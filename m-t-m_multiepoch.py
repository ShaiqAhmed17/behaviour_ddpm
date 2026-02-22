"""
Copy of ddpm/train/multiepoch.py that replaces task targets with samples
from an ablated source model.
"""

import argparse
import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from ddpm.model import BouncePopulationResidualModel

from purias_utils.util.arguments_yaml import ConfigNamepace

from ddpm.model.main.multiepoch import (
    MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess,
)
from ddpm import tasks, model
from ddpm.tasks.main.multiepoch import MultiEpochDiffusionTask
from ddpm.utils.vis import (
    plot_standard_losses_multiepoch,
    imshow_palimpsest_reprs,
    scatter_standard_responses_multiepoch,
    plot_mse_losses,
)

import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from purias_utils.util.logging import configure_logging_paths
from purias_utils.util.logging import LoopTimer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiepoch trainer with ablated-model targets"
    )
    parser.add_argument("config", type=str, help="Training config YAML")
    parser.add_argument(
        "--source_run_path",
        type=str,
        required=True,
        help="Run directory with args.yaml + checkpoint for target model",
    )
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default="state.mdl",
        help="Checkpoint filename inside source_run_path",
    )
    parser.add_argument(
        "--ablate_neuron",
        type=int,
        required=True,
        help="Neuron index to ablate in the source model (indexing the memory nullspace, not behavioral subspace)",
    )
    parser.add_argument(
        "--source_device",
        type=str,
        default=None,
        help="Device for source model (default: same as training device)",
    )
    parser.add_argument(
        "--noise_scaler",
        type=float,
        default=None,
        help="Noise scaler used when sampling from source model (default: use source model's trained setting)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of logging intervals without improvement before stopping (default: 5)",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.999,
        help="Relative improvement threshold for early stopping (default: 0.999 = 0.1%% improvement required)",
    )
    return parser.parse_args()


def load_model_from_run(run_path, device='cuda', checkpoint_name='state.mdl'):
    args_path = os.path.join(run_path, 'args.yaml')
    state_path = os.path.join(run_path, checkpoint_name)

    args = ConfigNamepace.from_yaml_path(args_path)

    task = getattr(tasks, args.task_name)(**args.task_config.dict)

    sigma2x_schedule = torch.linspace(
        args.starting_sigma2,
        args.ultimate_sigma2,
        args.num_timesteps,
    ).to(device)

    residual_model_kwargs = args.model_config.dict.pop("residual_model_kwargs").dict
    ddpm_model_kwargs = args.model_config.dict.pop("ddpm_model_kwargs").dict

    model_builder = getattr(model, args.model_name)

    try:
        prep_shape = task.sensory_gen.prep_sensory_shape
        underlying_shape = task.sensory_gen.underlying_sensory_shape
    except Exception:
        prep_shape = None
        underlying_shape = None

    try:
        sample_shape = task.distribution_gen.sample_shape
    except Exception:
        sample_shape = task.sample_gen.sample_shape

    ddpm_model, _, _ = model_builder(
        **args.model_config.dict,
        residual_model_kwargs=residual_model_kwargs,
        ddpm_model_kwargs=ddpm_model_kwargs,
        sigma2x_schedule=sigma2x_schedule,
        prep_sensory_shape=prep_shape,
        underlying_sensory_shape=underlying_shape,
        sample_shape=sample_shape,
        device=device,
    )

    ddpm_model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    ddpm_model.to(device)
    ddpm_model.eval()

    print(f"Loaded source model from {run_path}")
    return ddpm_model


def get_ablation_vector(model_instance, neuron_idx):
    """
    Extract the ablation vector from the model's nullspace.
    
    neuron_idx: index in the memory nullspace (0 to num_memory_neurons-1)
    
    Returns: normalized nullspace direction vector (shape: [ambient_dim])
    """
    # behaviour_nullspace has shape [num_memory_dims, ambient_dim]
    # Each row is an orthonormal basis vector for the nullspace
    print(f"\n[get_ablation_vector] Extracting nullspace direction {neuron_idx}")
    print(f"  behaviour_nullspace shape: {model_instance.behaviour_nullspace.shape}")
    
    ablation_vector = model_instance.behaviour_nullspace[neuron_idx].clone()
    print(f"  ablation_vector shape: {ablation_vector.shape}")
    print(f"  ablation_vector device: {ablation_vector.device}")
    
    # Should already be normalized from SVD, but ensure it
    initial_norm = torch.norm(ablation_vector).item()
    ablation_vector = ablation_vector / torch.norm(ablation_vector)
    final_norm = torch.norm(ablation_vector).item()
    
    print(f"  Initial norm: {initial_norm:.6f}, Final norm: {final_norm:.6f}")
    print(f"  First 5 elements: {ablation_vector[:5].cpu().numpy()}")
    
    return ablation_vector


def main():
    cli_args = parse_args()

    args = ConfigNamepace.from_yaml_path(cli_args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_device = cli_args.source_device or device

    # Unpack all args
    ultimate_sigma2 = args.ultimate_sigma2
    starting_sigma2 = args.starting_sigma2
    num_timesteps = args.num_timesteps
    num_samples = args.num_samples
    batch_size = args.batch_size
    num_trials = args.num_trials
    logging_freq = args.logging_freq
    save_base = args.save_base
    task_name = args.task_name
    task_config = args.task_config
    regularise_prep_state_weight = args.regularise_prep_state_weight
    regularise_prep_activity_indices = args.regularise_prep_activity_indices
    regularise_prep_activity_weight = args.regularise_prep_activity_weight
    regularise_diffusion_nullspace = args.regularise_diffusion_nullspace
    model_name = args.model_name
    model_config = args.model_config
    lr = args.lr
    resume_path = args.resume_path

    # Set up logging
    magma = plt.get_cmap("magma")
    cNorm = colors.Normalize(vmin=1, vmax=num_timesteps)
    kl_colors_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
    kl_colors_scalarMap.set_array([])
    timer = LoopTimer(num_trials)
    [training_print_path], save_base, _ = configure_logging_paths(
        save_base, log_suffixes=[f"train"], index_new=True
    )
    all_prep_state_losses = np.zeros([num_trials])
    all_delay_activity_losses = np.zeros([num_trials])
    args.write_to_yaml(os.path.join(save_base, "args.yaml"))

    # Generate noising schedule
    sigma2x_schedule = torch.linspace(starting_sigma2, ultimate_sigma2, num_timesteps)
    sigma2x_schedule = sigma2x_schedule.to(device=device)

    task: MultiEpochDiffusionTask = getattr(tasks, task_name)(**task_config.dict)
    task.save_metadata(os.path.join(save_base, "task_metadata"))
    try:
        all_individual_residual_mses = np.zeros([num_trials, task.sample_gen.limit_training_timesteps])
        plotting_num_timesteps = task.sample_gen.limit_training_timesteps
    except Exception:
        all_individual_residual_mses = np.zeros([num_trials, num_timesteps])
        plotting_num_timesteps = num_timesteps

    all_trial_type_trial_indices = {trial_type: [] for trial_type in task.task_variable_gen.trial_types}
    all_trial_type_trial_indices['all'] = []
    if len(task.task_variable_gen.trial_types) == 1:
        assert task.task_variable_gen.trial_types[0] == 'all'

    # Set up model
    residual_model_kwargs = model_config.dict.pop("residual_model_kwargs").dict
    ddpm_model_kwargs = model_config.dict.pop("ddpm_model_kwargs").dict
    ddpm_model, mse_key_pred, mse_key_target = getattr(model, model_name)(
        **model_config.dict,
        residual_model_kwargs=residual_model_kwargs,
        ddpm_model_kwargs=ddpm_model_kwargs,
        sigma2x_schedule=sigma2x_schedule,
        prep_sensory_shape=task.sensory_gen.prep_sensory_shape,
        underlying_sensory_shape=task.sensory_gen.underlying_sensory_shape,
        sample_shape=task.sample_gen.sample_shape,
        device=device,
    )

    ddpm_model: MultiPreparatoryLinearSubspaceTeacherForcedDDPMReverseProcess
    ddpm_model.to(device)

    # Set up training
    optim = torch.optim.Adam(ddpm_model.parameters(), lr=lr)

    if resume_path is not None:
        task.load_metadata(resume_path.replace('state.mdl', 'task_metadata.npy'))
        trained_state_dict = torch.load(resume_path, weights_only=True)
        if args.resume_kept_input_dims is not None:
            ddpm_model.load_state_dict(trained_state_dict, kept_input_dims=args.resume_kept_input_dims)
        else:
            try:
                ddpm_model.load_state_dict(trained_state_dict)
            except AssertionError as e:
                print(e)
                print('warning: loading state_dict non-strictly')
                ddpm_model.load_state_dict(trained_state_dict, strict=False)
                if isinstance(ddpm_model.residual_model, BouncePopulationResidualModel):
                    residual_model_state_dict = {k.removeprefix('residual_model.'): v for k, v in trained_state_dict.items() if k.startswith('residual_model.')}
                    ddpm_model.residual_model.load_state_dict(state_dict=residual_model_state_dict)
                    print('loaded BouncePopulationResidualModel from existing non bounce!')

        try:
            optim.load_state_dict(torch.load(resume_path.replace('state.mdl', 'opt_state.mdl'), weights_only=True))
        except Exception as e:
            print(e)

    # Source model for ablated targets
    source_model = load_model_from_run(
        cli_args.source_run_path,
        device=source_device,
        checkpoint_name=cli_args.source_checkpoint,
    )
    
    # Validate neuron index is in memory subspace range
    memory_dims = source_model.behaviour_nullspace.shape[0]
    if cli_args.ablate_neuron < 0 or cli_args.ablate_neuron >= memory_dims:
        raise ValueError(f"Neuron index {cli_args.ablate_neuron} out of range for memory subspace (0-{memory_dims-1})")
    
    # Extract ablation vector from nullspace (state-space ablation)
    ablation_vector = get_ablation_vector(source_model, cli_args.ablate_neuron)
    ablation_vector = ablation_vector.to(source_device)
    
    print(f"Source model has {memory_dims} nullspace basis vectors")
    print(f"Ablating nullspace direction {cli_args.ablate_neuron}")
    
    # CRITICAL: Copy projection matrices from source to training model
    # This ensures both models use the same behavioral/memory subspace decomposition
    print("Copying projection matrices from source to training model...")
    with torch.no_grad():
        ddpm_model.auxiliary_embedding_matrix.copy_(source_model.auxiliary_embedding_matrix.to(device))
        ddpm_model.behaviour_nullspace.copy_(source_model.behaviour_nullspace.to(device))
        ddpm_model.sample_subspace_accessor.copy_(source_model.sample_subspace_accessor.to(device))
        ddpm_model.behaviour_nullspace_accessor.copy_(source_model.behaviour_nullspace_accessor.to(device))

    # For transparency
    sch_fig, sch_axes = plt.subplots(2, 1, figsize=(5, 8))
    sch_axes[0].plot(
        ddpm_model.sigma2xt_schedule.cpu().numpy(), label="sigma2xt_schedule", alpha=0.4
    )
    sch_axes[0].plot(ddpm_model.a_t_schedule.cpu().numpy(), label="a_t_schedule", alpha=0.4)
    sch_axes[0].plot(
        ddpm_model.root_b_t_schedule.cpu().numpy(), label="root_b_t_schedule", alpha=0.4
    )
    sch_axes[0].plot(
        ddpm_model.noise_scaler_schedule.cpu().numpy(),
        label="noise_scaler_schedule",
        alpha=0.4,
    )
    sch_axes[0].plot(
        ddpm_model.base_samples_scaler_schedule.cpu().numpy(),
        label="base_samples_scaler_schedule",
        alpha=0.4,
    )
    sch_axes[0].plot(
        ddpm_model.residual_scaler_schedule.cpu().numpy(),
        label="residual_scaler_schedule",
        alpha=0.4,
    )
    sch_axes[0].legend()

    sch_axes[1].set_title("Time embeddings")
    sch_axes[1].imshow(ddpm_model.time_embeddings.time_embs.detach().cpu().numpy().T)

    plt.savefig(os.path.join(save_base, "sigma_schedule_unrolling.png"))

    # This will get filled in and continuously updated by task.sample_gen.generate_sample_diagnostics
    recent_sample_diagnostics = deque(maxlen=100)

    plotting_offset = 0
    plotting_start = 0
    
    # Early stopping tracking
    best_test_mse = float('inf')
    patience_counter = 0
    early_stopping_patience = cli_args.early_stopping_patience
    early_stopping_threshold = cli_args.early_stopping_threshold
    print(f"Early stopping enabled: patience={early_stopping_patience}, threshold={early_stopping_threshold}")

    for t in tqdm(range(num_trials)):

        prev_state = copy.deepcopy(ddpm_model.state_dict())

        timer.loop_start()

        trial_information = task.generate_trial_information(
            batch_size=batch_size, num_samples=num_samples
        )

        with torch.no_grad():
            # Move inputs to source device
            if isinstance(trial_information.prep_network_inputs, dict):
                prep_inputs_source = {k: v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in trial_information.prep_network_inputs.items()}
            else:
                prep_inputs_source = [v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                      for v in trial_information.prep_network_inputs]
            
            if isinstance(trial_information.diffusion_network_inputs, dict):
                diff_inputs_source = {k: v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in trial_information.diffusion_network_inputs.items()}
            else:
                diff_inputs_source = [v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                      for v in trial_information.diffusion_network_inputs]
            
            # Build kwargs for source model sampling
            source_sample_kwargs = {
                'prep_network_inputs': prep_inputs_source,
                'diffusion_network_inputs': diff_inputs_source,
                'prep_epoch_durations': trial_information.prep_epoch_durations,
                'diffusion_epoch_durations': trial_information.diffusion_epoch_durations,
                'samples_shape': [batch_size, num_samples],
                'ablation_vector': ablation_vector,  # Pass ablation vector for state-space ablation
            }
            if cli_args.noise_scaler is not None:
                source_sample_kwargs['noise_scaler'] = cli_args.noise_scaler
            
            source_samples_prep_dicts, source_samples_dict = source_model.generate_samples(**source_sample_kwargs)
            ablated_samples = source_samples_dict['samples'].to(device).float()

        trial_information.sample_information.sample_set = ablated_samples.detach().cpu()

        with torch.no_grad():
            forward_process = ddpm_model.noise(
                x_0=ablated_samples
            )
        preparatory_state_dicts, epsilon_hat_dict = ddpm_model.residual(
            x_samples=forward_process["x_t"],
            prep_network_inputs=trial_information.prep_network_inputs,
            diffusion_network_inputs=trial_information.diffusion_network_inputs,
            prep_epoch_durations=trial_information.prep_epoch_durations,
            diffusion_epoch_durations=trial_information.diffusion_epoch_durations,
        )
        residual_mse = task.sample_gen.mse(
            epsilon_hat_dict[mse_key_pred], forward_process[mse_key_target]
        )  # [batch, samples, time]

        total_loss = residual_mse.mean()

        prep_state_loss = (
            preparatory_state_dicts[-1]["postprep_base_samples"]
            .mean(-2)
            .square()
            .sum(-1)
            .sqrt()
            .mean(0)
        )  # [B,S,2] -> mean over samples [B,2] -> mag of mean [B] -> average of that <scalar>

        prep_activities_to_regularise = [preparatory_state_dicts[rpai]['preparatory_trajectory'] for rpai in regularise_prep_activity_indices]
        if regularise_diffusion_nullspace:
            prep_activities_to_regularise.append(epsilon_hat_dict["trajectories"] @ ddpm_model.behaviour_nullspace.T)

        delay_activity_loss = 0.0
        for patr in prep_activities_to_regularise:
            delay_activity_loss = delay_activity_loss + patr.square().sum(-1).sqrt().mean()

        total_loss = total_loss + (regularise_prep_state_weight * prep_state_loss) + (delay_activity_loss * regularise_prep_activity_weight)

        if total_loss.isnan() or total_loss.isinf():
            torch.save(prev_state, os.path.join(save_base, f"state_saved.mdl"))
            raise Exception(f'Total_loss goes to NaN or inf. {os.path.join(save_base, f"state_saved.mdl")} has latest safe state dict')

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if 'trial_type_idx' in trial_information.task_variable_information:
            import pdb; pdb.set_trace()
        else:
            all_trial_type_trial_indices['all'].append(t)

        if t % 100_000 == 0:
            torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.{t}.mdl"))
            torch.save(optim.state_dict(), os.path.join(save_base, f"opt_state.{t}.mdl"))

        if t >= plotting_start:

            if len(all_trial_type_trial_indices) == 1:
                assert 'trial_type_idx' not in trial_information.task_variable_information
            all_individual_residual_mses[t - plotting_start, :] = (
                residual_mse.detach().cpu().mean(0).mean(0)
            )

            all_prep_state_losses[t - plotting_start] = prep_state_loss.detach().cpu()
            all_delay_activity_losses[t - plotting_start] = delay_activity_loss.detach().cpu()

        if (t - plotting_offset) % logging_freq == 0:

            test_trial_information = task.generate_test_trial_information(num_samples=500)

            fig, axes = plt.subplots(
                1 + 2 * len(task.task_variable_gen.trial_types),
                7,
                figsize=(35, 5 * (1 + 2 * len(task.task_variable_gen.trial_types)))
            )

            plot_standard_losses_multiepoch(
                mse_ax=axes[0, 0], mean_mse_ax=axes[0, 1],
                zoomed_mse_ax=axes[0, 2], zoomed_mean_mse_ax=axes[0, 3],
                prep_state_reg_axes=axes[0, 4], delay_activity_reg_axes=axes[0, 5],
                training_step=t, plotting_start=plotting_start,
                diffusion_cmap=kl_colors_scalarMap,
                num_timesteps=plotting_num_timesteps,
                all_individual_residual_mses=all_individual_residual_mses[all_trial_type_trial_indices['all']],
                all_prep_state_losses=all_prep_state_losses,
                all_delay_activity_losses=all_delay_activity_losses,
                trial_type_name='all'
            )

            with torch.no_grad():
                # Generate test samples from ablated source model (the true training targets)
                if isinstance(test_trial_information.prep_network_inputs, dict):
                    test_prep_inputs_source = {k: v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                               for k, v in test_trial_information.prep_network_inputs.items()}
                else:
                    test_prep_inputs_source = [v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                               for v in test_trial_information.prep_network_inputs]
                
                if isinstance(test_trial_information.diffusion_network_inputs, dict):
                    test_diff_inputs_source = {k: v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                               for k, v in test_trial_information.diffusion_network_inputs.items()}
                else:
                    test_diff_inputs_source = [v.to(source_device) if isinstance(v, torch.Tensor) else v 
                                               for v in test_trial_information.diffusion_network_inputs]
                
                # Build kwargs for source model sampling
                source_test_kwargs = {
                    'prep_network_inputs': test_prep_inputs_source,
                    'diffusion_network_inputs': test_diff_inputs_source,
                    'prep_epoch_durations': test_trial_information.prep_epoch_durations,
                    'diffusion_epoch_durations': test_trial_information.diffusion_epoch_durations,
                    'samples_shape': [len(task.task_variable_gen.trial_types), 500],
                    'ablation_vector': ablation_vector,  # CRITICAL: Also ablate during test sampling
                }
                if cli_args.noise_scaler is not None:
                    source_test_kwargs['noise_scaler'] = cli_args.noise_scaler
                
                source_test_prep_dicts, source_test_samples_dict = source_model.generate_samples(**source_test_kwargs)
                ablated_test_samples = source_test_samples_dict['samples'].to(device).float()
                test_trial_information.sample_information.sample_set = ablated_test_samples.detach().cpu()
                
                test_forward_process = ddpm_model.noise(
                    x_0=ablated_test_samples
                )
                # Training model uses its own configured noise_scaler (not forced)
                novel_samples_prep_dicts, novel_samples_dict = ddpm_model.generate_samples(
                    prep_network_inputs=test_trial_information.prep_network_inputs,
                    diffusion_network_inputs=test_trial_information.diffusion_network_inputs,
                    prep_epoch_durations=test_trial_information.prep_epoch_durations,
                    diffusion_epoch_durations=test_trial_information.diffusion_epoch_durations,
                    samples_shape=[len(task.task_variable_gen.trial_types), 500],
                )
                
                # Compute test MSE for early stopping
                test_prep_dicts, test_epsilon_hat_dict = ddpm_model.residual(
                    x_samples=test_forward_process["x_t"],
                    prep_network_inputs=test_trial_information.prep_network_inputs,
                    diffusion_network_inputs=test_trial_information.diffusion_network_inputs,
                    prep_epoch_durations=test_trial_information.prep_epoch_durations,
                    diffusion_epoch_durations=test_trial_information.diffusion_epoch_durations,
                )
                test_residual_mse = task.sample_gen.mse(
                    test_epsilon_hat_dict[mse_key_pred], test_forward_process[mse_key_target]
                )
                current_test_mse = test_residual_mse.mean().item()
                
                # Early stopping logic
                if current_test_mse < best_test_mse * early_stopping_threshold:
                    # Improvement detected
                    improvement = (best_test_mse - current_test_mse) / best_test_mse * 100 if best_test_mse < float('inf') else 0
                    print(f"\n[t={t}] Test MSE improved: {best_test_mse:.6f} -> {current_test_mse:.6f} ({improvement:.2f}% improvement)")
                    best_test_mse = current_test_mse
                    patience_counter = 0
                    # Save best model
                    torch.save(ddpm_model.state_dict(), os.path.join(save_base, "best_state.mdl"))
                    torch.save(optim.state_dict(), os.path.join(save_base, "best_opt_state.mdl"))
                    print(f"Saved best model at trial {t}")
                else:
                    # No improvement
                    patience_counter += 1
                    print(f"\n[t={t}] Test MSE: {current_test_mse:.6f} (best: {best_test_mse:.6f}, patience: {patience_counter}/{early_stopping_patience})")

            for trial_type_idx, test_trial_type in enumerate(task.task_variable_gen.trial_types):

                trial_type_top_row_idx = trial_type_idx * 2 + 1

                scatter_standard_responses_multiepoch(
                    real_samples_axes=axes[trial_type_top_row_idx, 0],
                    generated_samples_axes=axes[trial_type_top_row_idx, 1],
                    early_predictions_axes=axes[trial_type_top_row_idx, 2],
                    start_of_prep_axes=axes[trial_type_top_row_idx, 3],
                    pred_residual_axes=axes[trial_type_top_row_idx, 4],
                    task=task,
                    task_variable_axes=[axes[trial_type_top_row_idx + 1, 0], axes[trial_type_top_row_idx + 1, 1]],
                    novel_samples_dict=novel_samples_dict,
                    novel_samples_prep_dicts=novel_samples_prep_dicts,
                    forward_process_dict=test_forward_process,
                    trial_info=test_trial_information,
                    diffusion_cmap=kl_colors_scalarMap,
                    trial_type_name=test_trial_type,
                    batch_idx=trial_type_idx
                )

                plot_mse_losses(
                    mse_ax=axes[trial_type_top_row_idx, 5],
                    mean_mse_ax=axes[trial_type_top_row_idx, 6],
                    zoomed_mse_ax=axes[trial_type_top_row_idx + 1, 5],
                    zoomed_mean_mse_ax=axes[trial_type_top_row_idx + 1, 6],
                    training_step=t,
                    plotting_start=plotting_start,
                    num_timesteps=plotting_num_timesteps,
                    diffusion_cmap=kl_colors_scalarMap,
                    trial_type_name=test_trial_type,
                    all_individual_residual_mses=all_individual_residual_mses[all_trial_type_trial_indices[test_trial_type]]
                )

                if 'palimpsest' in task_name:
                    stax, cax = [axes[trial_type_top_row_idx + 1, 2], axes[trial_type_top_row_idx + 1, 3]]
                    imshow_palimpsest_reprs([stax, cax], test_trial_information, task, [0, 2], trial_type_idx)

            fig.savefig(os.path.join(save_base, "latest_log.png"))

            plt.close(fig)

            torch.save(ddpm_model.state_dict(), os.path.join(save_base, f"state.mdl"))
            torch.save(optim.state_dict(), os.path.join(save_base, f"opt_state.mdl"))
            
            # Check early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING: No improvement for {early_stopping_patience} logging intervals")
                print(f"Best test MSE: {best_test_mse:.6f} at trial ~{t - patience_counter * logging_freq}")
                print(f"Final test MSE: {current_test_mse:.6f}")
                print(f"Best model saved at: {os.path.join(save_base, 'best_state.mdl')}")
                print(f"{'='*80}")
                break


if __name__ == "__main__":
    main()
