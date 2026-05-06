from typing import Literal

from ddpm.model.main.base.linear_subspace import *
from torch import Tensor as _T


class LinearSubspaceTeacherForcedHVAEReverseProcess(
    LinearSubspaceTeacherForcedDDPMReverseProcess
):
    """
    After the (re)rehaul on 27.03.2025

    Using the name "HVAE" to refer to models which directly generate samples from p(x_{t-1} | x_t)
    
    TODO: define the new functionality below at a more abstract level
        - e.g. class HVAEReverseProcessBase(DDPMReverseProcessBase): ...

    The main rehauls are that:
        1. denoise_one_step is not 'gated', we take in "predicted residual" and just add the transition kernelleakily
        2. noise also adds a target for the transition kernel
    """

    def __init__(self, *_, seperate_output_neurons: bool, stabilise_nullspace: bool, sample_ambient_dim: int, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, noise_scaler: float | Literal['nat'], train_as_rnn: bool, device="cuda", **kwargs) -> None:
        
        super().__init__(
            num_prep_steps=None,
            network_input_during_diffusion=None,
            seperate_output_neurons=seperate_output_neurons,
            stabilise_nullspace=stabilise_nullspace,
            sample_ambient_dim=sample_ambient_dim,
            sample_shape=sample_shape,
            sigma2xt_schedule=sigma2xt_schedule,
            residual_model=residual_model,
            input_model=input_model,
            time_embedding_size=time_embedding_size,
            device=device
        )

        assert self.stabilise_nullspace, "LinearSubspaceTeacherForcedDDPMReverseProcess requires stabilise_nullspace=True for full awareness! It will be removed later"
        del self.stabilise_nullspace    # Should not be used!
        
        self.train_as_rnn = train_as_rnn
        if train_as_rnn:
            self.do_teacher_forcing = False
        self.noise_scaler = noise_scaler
        
        if noise_scaler == 'nat':
            pass
        elif isinstance(noise_scaler, float):
            self.noise_scaler_schedule = (
                torch.ones_like(self.noise_scaler_schedule)
                * self.noise_scaler_schedule[1] * noise_scaler
            )
        else:
            raise ValueError(noise_scaler)

    def denoise_one_step(self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float, *_, override_euler_alpha: Optional[float] = None, ablation_vector: Optional[_T] = None):
        """
        predicted_residual (misnomer) now just acts as the integration term in continuous time, i.e. f(...) in

        \tau \dot x_t = -x_t + f(x_t, s_t, t) + v_t \eta

        using HVAE time convention => x_{t-1} = (1-euler_alpha) x_t + euler_alpha (f(x_t, s_t, t) + v_t \eta)

        ablation_vector: [D] to project out one direction, or [N, D] to project out N orthonormal directions.
        """
        euler_alpha = override_euler_alpha if override_euler_alpha is not None else self.euler_alpha

        noise = noise_scaler * torch.randn_like(x_t_plus_1)
        scaled_noise = noise * self.noise_scaler_schedule[-t_idx]

        leaky_term = (1 - euler_alpha) * x_t_plus_1
        integration_term = euler_alpha * predicted_residual

        x_t = leaky_term + integration_term + scaled_noise

        if ablation_vector is not None:
            if ablation_vector.dim() == 1:
                component = (x_t @ ablation_vector.unsqueeze(-1)) * ablation_vector
            else:
                component = (x_t @ ablation_vector.T) @ ablation_vector
            x_t = x_t - component

        fake_early_x0_pred = torch.ones_like(x_t) * torch.nan

        return x_t, fake_early_x0_pred
    
    def noise(self, x_0: _T) -> Dict[str, _T | int]:
        """
        Target for the transition kernel is based on the q_posterior mean:

            posterior_mean = mu_q = (x_t - \gamma_t \epsilon_t)
            \gamma_t = \beta_t / \sqrt{1-\bar\alpha_t}

        ...but altered by the euler discretisation:

            mu_q <-> (1-euler_alpha) x_t + euler_alpha f(x_t, s_t, t)

            ==> f(x_t, s_t, t) <-> (mu_q - (1-euler_alpha) x_t) / euler_alpha
        """
        noising_dict = super().noise(x_0)
        num_extra_dim = noising_dict['num_extra_dim']
        if self.train_as_rnn:
            noising_dict['unnoised_target'] = x_0.unsqueeze(num_extra_dim).expand(*noising_dict['x_t'].shape)
        else:
            scaled_base_samples = self.reshaped_base_samples_scaler_schedule[*[None]*num_extra_dim] * noising_dict['x_t']
            scaled_residual = self.reshaped_residual_scaler_schedule[*[None]*num_extra_dim] * noising_dict['epsilon']
            one_step_denoise_mean = scaled_base_samples - scaled_residual       # mu_q      [..., T, <shape samples>]
            
            try:
                kernel_target = (one_step_denoise_mean - (1 - self.euler_alpha) * noising_dict['x_t']) / self.euler_alpha
            except AttributeError:
                kernel_target = (one_step_denoise_mean - (1 - self.primary_euler_alpha) * noising_dict['x_t']) / self.primary_euler_alpha
            
            noising_dict['kernel_target'] = kernel_target
        return noising_dict





class PreparatoryHVAEReverseProcess(
    LinearSubspaceTeacherForcedHVAEReverseProcess,
    PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
):

    def __init__(self, *_, seperate_output_neurons: bool, stabilise_nullspace: bool, sample_ambient_dim: int, sample_shape: List[int], sigma2xt_schedule: _T, residual_model: VectoralResidualModel, input_model: InputModelBlock, time_embedding_size: int, noise_scaler: float, train_as_rnn: bool, device="cuda", **kwargs) -> None:
        super().__init__(
            num_prep_steps=None,
            network_input_during_diffusion=None,
            seperate_output_neurons=seperate_output_neurons,
            stabilise_nullspace=stabilise_nullspace,
            sample_ambient_dim=sample_ambient_dim,
            sample_shape=sample_shape,
            sigma2xt_schedule=sigma2xt_schedule,
            residual_model=residual_model,
            input_model=input_model,
            time_embedding_size=time_embedding_size,
            noise_scaler=noise_scaler,
            train_as_rnn = train_as_rnn,
            device=device
        )
