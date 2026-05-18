from typing import Literal

from ddpm.model.main.base.linear_subspace import *
from ddpm.model.main.base.hvae import *
from torch import Tensor as _T

from ddpm.model.transition import BouncePopulationResidualModel


class LinearSubspaceBounceNetworkTeacherForcedHVAEReverseProcess(
    LinearSubspaceTeacherForcedHVAEReverseProcess
):
    """
    29.03.2025

    Rather than rec(h) = W1 @ f(W2 @ f(h)) we will have two neuron populations
    The above case is recovered** when membrane time constant of the second population << that of primary population
        do this with bounce_euler_alpha = 1

    ** fineprint: no it's not quite recovered... TODO: explain this!
    """

    def __init__(
        self,
        *_,
        seperate_output_neurons: bool,
        primary_euler_alpha: float,
        bounce_euler_alpha: float,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: BouncePopulationResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        noise_scaler: float | Literal["nat"],
        bounce_noise_scalar: Optional[float],
        primary_noise_scalar: Optional[float],
        train_as_rnn: bool,
        device="cuda",
        **kwargs
    ) -> None:
        super().__init__(
            *_,
            seperate_output_neurons=seperate_output_neurons,
            stabilise_nullspace=True,
            sample_ambient_dim=sample_ambient_dim,
            sample_shape=sample_shape,
            sigma2xt_schedule=sigma2xt_schedule,
            residual_model=residual_model,
            input_model=input_model,
            time_embedding_size=time_embedding_size,
            noise_scaler=noise_scaler,
            train_as_rnn=train_as_rnn,
            device=device,
            **kwargs
        )

        del self.euler_alpha
        self.primary_euler_alpha = primary_euler_alpha
        self.bounce_euler_alpha = bounce_euler_alpha
        if bounce_noise_scalar is None:
            self.bounce_noise_scalar = 1.0
        else:
            self.bounce_noise_scalar = bounce_noise_scalar
        if primary_noise_scalar is None:
            self.primary_noise_scalar = 1.0
        else:
            self.primary_noise_scalar = primary_noise_scalar

        assert len(self.sample_ambient_dims) == 1
        self.sample_ambient_dims = [2, self.sample_ambient_dims[0]]

        assert len(residual_model.recurrence_hidden_layers) == 0
        assert residual_model.nonlin_first

    def denoise_one_step(
        self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float, *_, override_euler_alpha: Optional[float] = None, ablation_vector: Optional[_T] = None
    ):
        """
        Same as before, but for each population seperately
        This should make it clear why bounce_euler_alpha = 1' does not quite recover 2 layer nn case!
        """
        x_t_primary, x0_pred_primary = super().denoise_one_step(
            t_idx,
            x_t_plus_1[..., 0, :],
            predicted_residual[..., 0, :],
            noise_scaler * self.primary_noise_scalar,
            override_euler_alpha=self.primary_euler_alpha,
            ablation_vector=ablation_vector,
        )
        x_t_bounce, x0_pred_bounce = super().denoise_one_step(
            t_idx,
            x_t_plus_1[..., 1, :],
            predicted_residual[..., 1, :],
            noise_scaler * self.bounce_noise_scalar,
            override_euler_alpha=self.bounce_euler_alpha,
            ablation_vector=ablation_vector,
        )
        return torch.stack([x_t_primary, x_t_bounce], -2), torch.stack(
            [x0_pred_primary, x0_pred_bounce], -2
        )

    def tf_replace(self, one_step_denoising: _T, ts_embedded_samples) -> _T:
        """
        Teacher forcing only applied to primary population!

        one_step_denoising: [..., T = 1, P = 2, D]
        ts_embedded_samples: [..., T = 1, D]
        """

        # Course correct in the linear sample subspace ---> (new_one_step_denoising @ self.auxiliary_embedding_matrix.T).abs().max() is very small
        new_one_step_denoising = one_step_denoising
        new_one_step_denoising[..., 0, :] = new_one_step_denoising[..., 0, :] - (
            new_one_step_denoising[..., 0, :] @ self.sample_subspace_accessor
        )

        new_one_step_denoising[..., 0, :] = (
            new_one_step_denoising[..., 0, :] + ts_embedded_samples
        )

        return new_one_step_denoising

    def extract_subspace(self, embedded_information: _T) -> _T:
        assert embedded_information.shape[-2] == 2
        return super(
            LinearSubspaceBounceNetworkTeacherForcedHVAEReverseProcess, self
        ).extract_subspace(embedded_information[..., 0, :])


class PreparatoryBounceNetworkHVAEReverseProcess(
    LinearSubspaceBounceNetworkTeacherForcedHVAEReverseProcess,
    PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
):

    def __init__(
        self,
        *_,
        seperate_output_neurons: bool,
        primary_euler_alpha: float,
        bounce_euler_alpha: float,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        noise_scaler: float,
        bounce_noise_scalar: Optional[float],
        primary_noise_scalar: Optional[float],
        train_as_rnn: bool,
        device="cuda",
        **kwargs
    ) -> None:
        super().__init__(
            num_prep_steps=None,
            network_input_during_diffusion=None,
            primary_euler_alpha=primary_euler_alpha,
            bounce_euler_alpha=bounce_euler_alpha,
            seperate_output_neurons=seperate_output_neurons,
            sample_ambient_dim=sample_ambient_dim,
            sample_shape=sample_shape,
            sigma2xt_schedule=sigma2xt_schedule,
            residual_model=residual_model,
            input_model=input_model,
            time_embedding_size=time_embedding_size,
            noise_scaler=noise_scaler,
            bounce_noise_scalar=bounce_noise_scalar,
            primary_noise_scalar=primary_noise_scalar,
            train_as_rnn=train_as_rnn,
            device=device,
        )
