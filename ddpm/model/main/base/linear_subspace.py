from ddpm.model.main.base.basic import *
from torch import Tensor as _T


class LinearSubspaceTeacherForcedDDPMReverseProcess(
    TeacherForcedDDPMReverseProcessBase
):
    """
    A special case where the sample space is embedded as a linear subspace, with the remaining directions
    used as the auxiliary computation variables

    Generates its own buffer auxiliary_embedding_matrix, which should not be confused with VectoralEmbeddedExampleSampleGenerator.linking_matrix!
    """

    def __init__(
        self,
        *_,
        seperate_output_neurons: bool,
        stabilise_nullspace: bool,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device="cuda",
    ) -> None:
        super().__init__(
            sample_shape,
            sigma2xt_schedule,
            residual_model,
            input_model,
            time_embedding_size,
            device,
        )

        self.do_teacher_forcing = True

        assert len(sample_shape) == 1 and (sample_ambient_dim >= sample_shape[0])
        # assert isinstance(self.residual_model, VectoralResidualModel) --> can be BouncePopulationResidualModel now!
        assert self.residual_model.state_space_size == sample_ambient_dim

        self.sample_ambient_dims = [sample_ambient_dim]

        self.stabilise_nullspace = stabilise_nullspace
        if stabilise_nullspace:
            self.euler_alpha = 0.1

        if sample_ambient_dim > sample_shape[0]:
            if seperate_output_neurons:
                orth = torch.eye(sample_ambient_dim)
            else:
                gaus = torch.randn(sample_ambient_dim, sample_ambient_dim)
                svd = torch.linalg.svd(gaus)
                orth = svd[0] @ svd[2]
        else:
            orth = torch.eye(sample_shape[0])
        self.register_buffer(
            "auxiliary_embedding_matrix", orth[: sample_shape[0]]
        )  # [little space, big space]
        self.register_buffer(
            "sample_subspace_accessor",
            orth[: sample_shape[0]].T @ orth[: sample_shape[0]],
        )  # [big space, big space]
        self.register_buffer(
            "behaviour_nullspace", orth[sample_shape[0] :]
        )  # [extra dims, big space]
        self.register_buffer(
            "behaviour_nullspace_accessor",
            orth[sample_shape[0] :].T @ orth[sample_shape[0] :],
        )  # [big space, big space]

    def denoise_one_step(
        self, t_idx: int, x_t_plus_1: _T, predicted_residual: _T, noise_scaler: float
    ):
        if self.stabilise_nullspace:
            coeff = (
                1 - self.euler_alpha - self.base_samples_scaler_schedule[-t_idx]
            ) / self.residual_scaler_schedule[-t_idx]
            stabilising_correction = coeff * (
                x_t_plus_1 @ self.behaviour_nullspace_accessor
            )
            predicted_residual = predicted_residual - stabilising_correction
        return super().denoise_one_step(
            t_idx, x_t_plus_1, predicted_residual, noise_scaler
        )


    def extract_subspace(self, embedded_information: _T) -> _T:
        return embedded_information @ self.auxiliary_embedding_matrix.T


    def tf_init(self, x_samples: _T) -> _T:
        return (
            x_samples @ self.auxiliary_embedding_matrix
        )  # [..., T, ambient space dim] --> will be used for teacher forcing
    
    def tf_replace(self, one_step_denoising: _T, ts_embedded_samples) -> _T:
        # Course correct in the linear sample subspace ---> (sample_removed_one_step_denoising @ self.auxiliary_embedding_matrix.T).abs().max() is very small
        sample_removed_one_step_denoising = one_step_denoising - (
            one_step_denoising @ self.sample_subspace_accessor
        )
        # NB: this indexing will have to be fixed for structed data...
        new_one_step_denoising = (
            sample_removed_one_step_denoising
            + ts_embedded_samples
        )
        return new_one_step_denoising

    def residual(
        self,
        x_samples: _T,
        network_input: _T,
        initial_state: Optional[_T] = None,
        kwargs_for_residual_model={},
    ) -> Dict[str, _T]:
        """
        x_samples of shape [..., T, <shape x>]
            Importantly, these now define continuous, gradually noised sample trajectories, not one-shot noising from the GT samples

        network_input of shape [..., <shape Z>]
            still the same for all timesteps, if the network is to be used for computation beforehand, it can should do so
            before calling this method, i.e. encoded in initial_state - see below

        initial_state of shape [..., ambient space dim]
            starting combined state of sample variable and auxiliary computation variables, all embedded into the larger ambient space

        Key differences here compared to OneShotDDPMReverseProcess:
            - Embedding: the x_samples are embedded into the ambient space before passing it throught the residual model prediction,
                then extracted to give epsilon_hat, as we only train on that linear subspace
            - Teacher-forcing: epsilon_hat prediction is unfortunately no longer parallelised across timesteps, but is instead done autoregressively.
                Furthermore, at each step, the sample subspace of the ambient space is subtracted, and instead the real x_samples value is added in its place.
        """
        num_extra_dim = (
            len(x_samples.shape) - len(self.sample_shape) - 1
        )  # including time now
        batch_shape = x_samples.shape[:num_extra_dim]
        input_vectors: _T = self.input_model(network_input, self.T)

        assert tuple(input_vectors.shape) == (
            *batch_shape,
            self.T,
            self.residual_model.input_size,
        ), f"Expected input_vector shape to be {(*batch_shape, self.T, self.residual_model.input_size)} but got {tuple(input_vectors.shape)}"
        assert tuple(x_samples.shape) == (
            *batch_shape,
            self.T,
            *self.sample_shape,
        ), f"Expected x_samples shape to end in {self.sample_shape} but got {x_samples.shape}"

        t_embeddings = self.time_embeddings(self.t_schedule)
        if self.do_teacher_forcing:            
            embedded_samples = self.tf_init(x_samples)

        if initial_state is None:
            initial_state = (
                torch.randn(
                    *batch_shape,
                    *self.sample_ambient_dims,
                    device=self.sigma2xt_schedule.device,
                )
                * self.base_std
            )
        else:
            assert tuple(initial_state.shape) == (
                *batch_shape,
                *self.sample_ambient_dims,
            ), f"Expected initial_state shape to end with {self.sample_ambient_dims} but got {tuple(initial_state.shape)}"

        one_step_denoising = initial_state.unsqueeze(-len(self.sample_ambient_dims)-1)  # [..., 1, ambient space dim]
        all_predicted_residuals = []
        all_subspace_trajectories = []
        all_trajectories = []

        for t_idx in range(1, self.T + 1):

            # If required, correct the state. This is equivalent to just replacing the embedded_predicted_residual
            if self.do_teacher_forcing:
                one_step_denoising = self.tf_replace(one_step_denoising, embedded_samples[..., [-t_idx], :])

            # Denoise in the full ambient space for one step: one_step_denoising, early_embedded_x0_pred both of shape [..., 1, ambient space dim]
            t_embedding = t_embeddings[-t_idx][None]

            embedded_predicted_residual = self.residual_model(
                one_step_denoising, t_embedding, input_vectors[..., [-t_idx], :]
            )
            one_step_denoising, early_embedded_x0_pred = self.denoise_one_step(
                t_idx, one_step_denoising, embedded_predicted_residual, noise_scaler=1.0
            )

            # Only these directions have the actual interpretation of a 'predicted residual'
            predicted_residual = self.extract_subspace(embedded_predicted_residual)
            all_predicted_residuals.append(predicted_residual)

            subspace_activity = self.extract_subspace(one_step_denoising)
            all_subspace_trajectories.append(subspace_activity)
            
            all_trajectories.append(one_step_denoising)
        
        epsilon_hat = torch.concat(
            all_predicted_residuals[::-1], num_extra_dim
        )  # forward (diffusion) time for downstream MSE loss!
        assert x_samples.shape == epsilon_hat.shape

        subspace_trajectories = torch.concat(
            all_subspace_trajectories, num_extra_dim
        )  # keep as reverse (denoising) time!
        assert x_samples.shape == subspace_trajectories.shape

        trajectories = torch.concat(
            all_trajectories, num_extra_dim
        )

        return {
            "epsilon_hat": epsilon_hat,                         # [..., T, Dx]
            "subspace_trajectories": subspace_trajectories,     # [..., T, Dx]
            "trajectories": trajectories,                       # [..., T, <shape z>]
        }

    def generate_samples(
        self,
        # target_distribution: DistributionInformation,   # For debugging only!
        *_,
        network_input: _T,
        samples_shape: Optional[List[int]] = None,
        base_samples: Optional[_T] = None,
        noise_scaler: float = 1.0,
        kwargs_for_residual_model={},
        start_t_idx=1,
        end_t_idx=None,
        ablation_vector: Optional[_T] = None,
    ) -> Dict[str, _T]:
        """
        Only difference to OneShotDDPMReverseProcess.generate_samples is that the denoising is done in the ambient space, not in the
            sample space. Samples are decoded at the end

        input_vector of shape [..., <shape Z>] purely out of convinience for us!
        If provided, base_samples of shape [..., ambient space dim] !!!
        Otherwise, give B = num_samples

        sample_trajectory of shape [..., T, <shape x>]
        samples of shape [..., <shape x>]
        early_x0_preds of shape [..., T, <shape x>]
        """
        if end_t_idx is None:
            end_t_idx = self.T
        assert (
            (1 <= start_t_idx) and (start_t_idx <= end_t_idx) and (end_t_idx <= self.T)
        )
        num_timesteps = end_t_idx - start_t_idx + 1

        assert (samples_shape is None) != (base_samples is None)

        if base_samples is None:
            base_samples = (
                torch.randn(
                    *samples_shape,
                    *self.sample_ambient_dims,
                    device=self.sigma2xt_schedule.device,
                )
                * self.base_std
            )
        else:
            samples_shape = base_samples.shape[:-len(self.sample_ambient_dims)]
            assert tuple(base_samples.shape) == (
                *samples_shape,
                *self.sample_ambient_dims,
            )

        input_vectors = self.input_model(network_input, num_timesteps)
        assert tuple(input_vectors.shape) == (
            *samples_shape,
            num_timesteps,
            self.residual_model.input_size,
        ), f"Expected input_vector shape to be {(*samples_shape, num_timesteps, *self.residual_model.input_size)} but got {tuple(input_vectors.shape)}"

        base_samples = base_samples.unsqueeze(len(samples_shape))  # [..., 1, D]
        t_embeddings = self.time_embeddings(self.t_schedule)

        embedded_sample_trajectory = []
        early_x0_preds = []
        all_predicted_residual = []

        for t_idx in range(start_t_idx, end_t_idx + 1):

            t_embedding = t_embeddings[-t_idx][None]

            predicted_residual = self.residual_model(
                base_samples, t_embedding, input_vectors[..., [-t_idx], :]
            )

            base_samples, early_embedded_x0_pred = self.denoise_one_step(
                t_idx, base_samples, predicted_residual, noise_scaler, ablation_vector=ablation_vector
            )
            
            early_x0_pred = self.extract_subspace(early_embedded_x0_pred) # [..., 1, sample dim]

            embedded_sample_trajectory.append(base_samples.detach())
            early_x0_preds.append(early_x0_pred.detach())
            all_predicted_residual.append(predicted_residual.detach())

        embedded_sample_trajectory = torch.concat(
            embedded_sample_trajectory, -len(self.sample_ambient_dims)-1
        )  # [..., T, sample_ambient_dim]


        sample_trajectory = self.extract_subspace(embedded_sample_trajectory)   # [..., T, dim x]
        early_x0_preds = torch.concat(early_x0_preds, -2)  # [..., T, dim x]

        all_predicted_residual = self.extract_subspace(torch.concat(all_predicted_residual, -len(self.sample_ambient_dims)-1))
        new_samples = self.extract_subspace(base_samples.squeeze(-len(self.sample_ambient_dims)-1).detach())
        
        return {
            "end_state": base_samples.squeeze(len(samples_shape)),
            "sample_trajectory": sample_trajectory.cpu(),
            "embedded_sample_trajectory": embedded_sample_trajectory.cpu(),
            "samples": new_samples.detach().cpu(),
            "early_x0_preds": early_x0_preds.cpu(),
            "epsilon_hat": all_predicted_residual.detach().cpu(),
        }



class PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess(
    LinearSubspaceTeacherForcedDDPMReverseProcess
):
    """
    This one is designed for tasks which do *not* have multiple epochs

    We are 'artificially' adding a preparatory time, during which we give the network the same stimulus they would receive anyway
    During the diffusion time, we have the option of no stimulus or the same stimulus

    This is why preparatory steps and diffusion-time input masking are given as class arguments = num_prep_steps, network_input_during_diffusion
    """

    def __init__(
        self,
        *_,
        num_prep_steps: int,
        network_input_during_diffusion: bool,
        seperate_output_neurons: bool,
        stabilise_nullspace: bool,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device="cuda",
        **kwargs
    ) -> None:

        super().__init__(
            seperate_output_neurons=seperate_output_neurons,
            stabilise_nullspace=stabilise_nullspace,
            sample_ambient_dim=sample_ambient_dim,
            sample_shape=sample_shape,
            sigma2xt_schedule=sigma2xt_schedule,
            residual_model=residual_model,
            input_model=input_model,
            time_embedding_size=time_embedding_size,
            device=device,
        )

        self.register_parameter(
            "prep_time_embedding",
            nn.Parameter(torch.randn(1, time_embedding_size), requires_grad=True),
        )
        self.network_input_during_diffusion = network_input_during_diffusion
        self.num_prep_steps = num_prep_steps

    def prepare(
        self,
        network_input: _T,
        batch_shape: List[int],
        num_steps: int,
        kwargs_for_residual_model={},
        *_,
        noise_scaler: float = 1.0,
        override_initial_state: Optional[_T] = None,
        ablation_vector: Optional[_T] = None,
    ) -> Dict[str, _T]:
        """
        Generate initial states to feed into residual or generate_samples

        As with those methods, input_vector of shape [..., <shape Z>], where [...] given by batch_shape
            Won't check this here as they will be checked downstream
        """
        if override_initial_state is None:
            initial_state = (
                torch.randn(
                    *batch_shape,
                    1,
                    *self.sample_ambient_dims,
                    device=self.sigma2xt_schedule.device,
                    dtype=self.sigma2xt_schedule.dtype,
                ).abs()
                * self.base_std
            )  # [..., 1, D]
        else:
            assert override_initial_state.shape == tuple(
                [*batch_shape, *self.sample_ambient_dims]
            )
            initial_state = override_initial_state.unsqueeze(-len(self.sample_ambient_dims)-1)

        input_vectors = self.input_model(network_input, num_steps)

        recent_state = initial_state
        preparatory_trajectory = []

        for t_idx in range(num_steps):

            # NB: this is not actually a residual!
            embedded_predicted_residual = self.residual_model(
                recent_state, self.prep_time_embedding, input_vectors[..., [-t_idx], :]
            )
            recent_state, _ = self.denoise_one_step(
                1, recent_state, embedded_predicted_residual, noise_scaler=noise_scaler
            )
            
            if ablation_vector is not None:
                if ablation_vector.dim() == 1:
                    component = (recent_state @ ablation_vector.unsqueeze(-1)) * ablation_vector
                else:
                    component = (recent_state @ ablation_vector.T) @ ablation_vector
                recent_state = recent_state - component
            
            preparatory_trajectory.append(recent_state)
        
        preparatory_trajectory = torch.concat(preparatory_trajectory, len(batch_shape))  # Reverse time!
        last_preparatory_trajectory_slice = preparatory_trajectory[*[slice(None) for _ in batch_shape], -1, :]

        return {
            "preparatory_trajectory": preparatory_trajectory,
            "postprep_state": last_preparatory_trajectory_slice,  # Again, reverse time, so final state here will be first state for the denoising
            "postprep_base_samples": self.extract_subspace(last_preparatory_trajectory_slice)      
        }

    def residual(
        self, x_samples: _T, network_input: _T, kwargs_for_residual_model={}
    ) -> Dict[str, _T]:
        prep_dict = self.prepare(
            network_input=network_input,
            batch_shape=x_samples.shape[:-2],
            num_steps=self.num_prep_steps,
        )
        network_input_mult = 1.0 if self.network_input_during_diffusion else 0.0
        residual_dict = super().residual(
            x_samples,
            network_input * network_input_mult,
            prep_dict["postprep_state"],
            kwargs_for_residual_model,
        )
        return dict(**prep_dict, **residual_dict)

    def generate_samples(
        self,
        *_,
        network_input: _T,
        samples_shape: List[int],
        noise_scaler: float = 1.0,
        kwargs_for_residual_model={},
        end_t_idx=None,
    ) -> Dict[str, _T]:
        prep_dict = self.prepare(
            network_input=network_input,
            batch_shape=samples_shape,
            num_steps=self.num_prep_steps,
        )
        network_input_mult = 1.0 if self.network_input_during_diffusion else 0.0
        samples_dict = super().generate_samples(
            network_input=network_input * network_input_mult,
            samples_shape=None,
            base_samples=prep_dict["postprep_state"],
            noise_scaler=noise_scaler,
            kwargs_for_residual_model=kwargs_for_residual_model,
            start_t_idx=1,  # Allowing a start t_idx doesn't really make sense if you have to do preparation
            end_t_idx=end_t_idx,
        )
        return dict(**prep_dict, **samples_dict)



class RNNBaselineDDPMReverseProcess(LinearSubspaceTeacherForcedDDPMReverseProcess):
    """
    Just a baseline, noiseless baseline to see if the dynamics of the network are good
    enough to do the most basic task(s) - e.g. a unimodal distribution

    Generation is the same as LinearSubspaceTeacherForcedDDPMReverseProcess, since no teacher-forcing is done there,
        but residual estimation is a bit of hack - it's not actually residual estimation - see below!

    To go even even closer to the baseline, use_leaky will abandon the standard scaling of noise, old state, and non-linear
        output in favour of the standard leaky RNN dynamics, eith euler alpha = 0.1
    """

    def __init__(
        self,
        *_,
        seperate_output_neurons: bool,
        use_leaky: bool,
        stabilise_nullspace: bool,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device="cuda",
        **kwargs
    ) -> None:
        super(RNNBaselineDDPMReverseProcess, self).__init__(
            seperate_output_neurons = seperate_output_neurons,
            stabilise_nullspace = stabilise_nullspace,
            sample_ambient_dim = sample_ambient_dim,
            sample_shape = sample_shape,
            sigma2xt_schedule = sigma2xt_schedule,
            residual_model = residual_model,
            input_model = input_model,
            time_embedding_size = time_embedding_size,
            device = device,
            **kwargs
        )

        self.do_teacher_forcing = False

        assert not (stabilise_nullspace and use_leaky)

        if use_leaky:
            self.noise_scaler_schedule[1:] = (
                torch.ones_like(self.noise_scaler_schedule[1:])
                * self.noise_scaler_schedule[1]
            )
            self.base_samples_scaler_schedule = (
                torch.ones_like(self.base_samples_scaler_schedule) * 0.9
            )  # like 1 - euler_alpha
            self.residual_scaler_schedule = 1.0 - self.base_samples_scaler_schedule

    def noise(self, x_0: _T) -> Dict[str, _T | int]:
        """
        x_0 of shape [..., <shape x>]

        **Slightly hacking the system here**

            x_t = nothing - no changes made to x_0 except to repeat on time dimension --> [..., T, <shape x>]
            epsilon = same as x_t!!! This will be fed to ExampleSampleGenerator.mse alongside the outputs of
                the RNNBaselineDDPMReverseProcess.residual, and in this case we just do vanilla mse
        """
        assert (
            list(x_0.shape[-len(self.sample_shape) :]) == self.sample_shape
        ), f"Expected samples that end with shape {self.sample_shape}, got samples of shape {x_0.shape}"

        num_extra_dim = len(x_0.shape) - len(self.sample_shape)
        x_t = x_0.unsqueeze(-len(self.sample_shape) - 1)
        x_t = x_t.expand(*x_0.shape[:num_extra_dim], self.T, *self.sample_shape)

        return {
            "x_t": x_t,
            "epsilon": x_t,
        }

    def residual(
        self,
        x_samples: _T,
        network_input: _T,
        initial_state: Optional[_T] = None,
        kwargs_for_residual_model={},
    ) -> Dict[str, _T]:
        """
        **Again, hacking the system here**

        x_samples of shape [..., T, <shape x>] but not used at all here!
        network_input of shape [..., <shape Z>] - as in LinearSubspaceTeacherForcedDDPMReverseProcess
        initial_state of shape [..., ambient space dim] - as in LinearSubspaceTeacherForcedDDPMReverseProcess

        This is the RNN baseline class:
            - No teacher forcing
            - output no longer predicts residuals (epsilon_hat above) but gives the network trajectory in the linear subspace of the sames
                NB: only intended to be used for the simplest tasks!
        """
        super(RNNBaselineDDPMReverseProcess, self).residual(
            x_samples=x_samples,
            network_input=network_input,
            initial_state=initial_state,
            **kwargs_for_residual_model,
        )



class PreparatoryRNNBaselineDDPMReverseProcess(
    RNNBaselineDDPMReverseProcess,
    PreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
):
    def __init__(
        self,
        *_,
        num_prep_steps: int,
        network_input_during_diffusion: bool,
        seperate_output_neurons: bool,
        use_leaky: bool,
        stabilise_nullspace: bool,
        sample_ambient_dim: int,
        sample_shape: List[int],
        sigma2xt_schedule: _T,
        residual_model: VectoralResidualModel,
        input_model: InputModelBlock,
        time_embedding_size: int,
        device="cuda",
        **kwargs
    ) -> None:
        super().__init__(
            num_prep_steps = num_prep_steps,
            network_input_during_diffusion = network_input_during_diffusion,
            seperate_output_neurons = seperate_output_neurons,
            use_leaky = use_leaky,
            stabilise_nullspace = stabilise_nullspace,
            sample_ambient_dim = sample_ambient_dim,
            sample_shape = sample_shape,
            sigma2xt_schedule = sigma2xt_schedule,
            residual_model = residual_model,
            input_model = input_model,
            time_embedding_size = time_embedding_size,
            device = device,
        )

