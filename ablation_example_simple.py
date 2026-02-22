"""
Simple Neuron Ablation Example

This is a streamlined example showing how to:
1. Load a trained model
2. Ablate a neuron
3. Compare outputs
4. Perform model-to-model recovery

Usage:
    python ablation_example_simple.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from types import SimpleNamespace

# Assuming your project structure
from purias_utils.util.arguments_yaml import ConfigNamepace
from ddpm import tasks, model as model_module


def load_model_simple(run_path: str, device='cuda'):
    """
    Simple model loading from a run directory.
    
    Args:
        run_path: Path to run directory (e.g., 'results_link_sampler/my_run')
        device: Device to load on
    
    Returns:
        model, task, args
    """
    args_path = os.path.join(run_path, 'args.yaml')
    state_path = os.path.join(run_path, 'state.mdl')
    
    # Load config
    args = ConfigNamepace.from_yaml_path(args_path)
    
    # Build task
    task = getattr(tasks, args.task_name)(**args.task_config.dict)
    
    # Build model
    sigma2x_schedule = torch.linspace(
        args.starting_sigma2, 
        args.ultimate_sigma2, 
        args.num_timesteps
    ).to(device)
    
    residual_model_kwargs = args.model_config.dict.pop("residual_model_kwargs").dict
    ddpm_model_kwargs = args.model_config.dict.pop("ddpm_model_kwargs").dict
    
    model_builder = getattr(model_module, args.model_name)
    
    # Try to get the right shape attributes from task
    try:
        prep_shape = task.sensory_gen.prep_sensory_shape
        underlying_shape = task.sensory_gen.underlying_sensory_shape
    except:
        prep_shape = None
        underlying_shape = None
    
    try:
        sample_shape = task.distribution_gen.sample_shape
    except:
        sample_shape = task.sample_gen.sample_shape
    
    ddpm_model, mse_key_pred, mse_key_target = model_builder(
        **args.model_config.dict,
        residual_model_kwargs=residual_model_kwargs,
        ddpm_model_kwargs=ddpm_model_kwargs,
        sigma2x_schedule=sigma2x_schedule,
        prep_sensory_shape=prep_shape,
        underlying_sensory_shape=underlying_shape,
        sample_shape=sample_shape,
        device=device,
    )
    
    # Load weights
    ddpm_model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    ddpm_model.to(device)
    ddpm_model.eval()
    
    print(f"✓ Loaded model from {run_path}")
    return ddpm_model, task, args


def ablate_neuron_simple(model, neuron_idx: int):
    """
    Ablate a neuron by zeroing its weights.
    
    This works for VectoralResidualModel and similar architectures.
    """
    print(f"\nAblating neuron {neuron_idx}...")
    
    residual_model = model.residual_model
    
    # Store original weights (optional, for comparison)
    original_state = {k: v.clone() for k, v in residual_model.state_dict().items()}

    # VectoralResidualModel: zero out weights in all linear layers
    if hasattr(residual_model, "layers"):
        for layer in residual_model.layers:
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    # Zero incoming weights (column)
                    if neuron_idx < layer.weight.shape[1]:
                        layer.weight.data[:, neuron_idx] = 0.0
                    
                    # Zero outgoing weights (row)
                    if neuron_idx < layer.weight.shape[0]:
                        layer.weight.data[neuron_idx, :] = 0.0
                        if layer.bias is not None:
                            layer.bias.data[neuron_idx] = 0.0
                            
    # DendriticResidualModel: zero weights for the neuron across the dendritic tree
    elif hasattr(residual_model, "axonal_tree"):
        import math

        for idx, tree_layer in enumerate(residual_model.axonal_tree):
            # Fanout node: maps from neurons -> neuron * branching
            if hasattr(tree_layer, "axon_weights"):
                num_neurons = tree_layer.num_neurons
                if neuron_idx >= num_neurons:
                    continue
                block = math.prod(tree_layer.branching_factors)
                row_start = neuron_idx * block
                row_end = row_start + block
                with torch.no_grad():
                    # Zero incoming contribution from the neuron
                    tree_layer.axon_weights.weight.data[:, neuron_idx] = 0.0

                    # Zero outgoing block for this neuron across fanout
                    tree_layer.axon_weights.weight.data[row_start:row_end, :] = 0.0
                    if tree_layer.axon_weights.bias is not None:
                        tree_layer.axon_weights.bias.data[row_start:row_end] = 0.0

                    # Ensure no time/input contributions for this neuron's block
                    tree_layer.time_weights.weight.data[row_start:row_end, :] = 0.0
                    tree_layer.input_weights.weight.data[row_start:row_end, :] = 0.0
            # Integration node: reduces branching -> neuron activation
            elif hasattr(tree_layer, "propagation_weights"):
                num_neurons = tree_layer.num_neurons
                if neuron_idx >= num_neurons:
                    continue
                with torch.no_grad():
                    tree_layer.propagation_weights.data[neuron_idx, ...] = 0.0

                    block = math.prod(tree_layer.branching_factors[:-1])
                    row_start = neuron_idx * block
                    row_end = row_start + block
                    tree_layer.time_representation_weights.weight.data[row_start:row_end, :] = 0.0
                    if tree_layer.time_representation_weights.bias is not None:
                        tree_layer.time_representation_weights.bias.data[row_start:row_end] = 0.0
    else:
        raise AttributeError(
            f"Unsupported residual model type: {type(residual_model).__name__}"
        )
    
    print(f"✓ Neuron {neuron_idx} ablated")
    return original_state


def generate_samples_batch(model, task, num_batches=10, batch_size=512, device='cuda'):
    """Generate samples from the model."""
    all_samples = []
    
    print(f"Generating {num_batches} batches...")
    model.eval()
    
    with torch.no_grad():
        for i in range(num_batches):
            # Use the task's trial API so we stay consistent with training
            trial_info = task.generate_trial_information(
                batch_size=1,
                num_samples=batch_size,
            )

            # Multi-epoch tasks provide prep + diffusion inputs and durations
            if hasattr(trial_info, 'prep_network_inputs'):
                _, samples_dict = model.generate_samples(
                    prep_network_inputs=trial_info.prep_network_inputs,
                    diffusion_network_inputs=trial_info.diffusion_network_inputs,
                    prep_epoch_durations=trial_info.prep_epoch_durations,
                    diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
                    samples_shape=[1, batch_size],
                    noise_scaler=1.0,
                )
            else:
                # Single-epoch tasks use network_inputs directly
                samples_dict = model.generate_samples(
                    network_input=trial_info.network_inputs[[0], :].to(device),
                    samples_shape=[1, batch_size],
                    noise_scaler=1.0,
                )
            
            samples = samples_dict['samples']
            if samples.shape[0] == 1:
                samples = samples.squeeze(0)
            all_samples.append(samples.cpu())
            
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{num_batches} batches complete")
    
    all_samples = torch.cat(all_samples, dim=0)
    print(f"✓ Generated {all_samples.shape[0]} samples")
    return all_samples


def build_trial_information_for_fixed_tasks(task, task_variable_information, num_samples: int):
    """Build trial information using a fixed task variable dict for all models."""
    try:
        return task.generate_trial_information(
            batch_size=None,
            num_samples=num_samples,
            override_task_variable_information=task_variable_information,
        )
    except TypeError:
        network_inputs = task.sensory_gen.generate_sensory_inputs(
            task_variable_information
        )
        network_inputs = network_inputs.unsqueeze(1).expand(
            task_variable_information[list(task_variable_information.keys())[0]].shape[0],
            num_samples,
            *task.sensory_gen.sensory_shape,
        )
        sample_information = task.sample_gen.generate_sample_set(
            num_samples, task_variable_information
        )
        return SimpleNamespace(
            task_variable_information=task_variable_information,
            network_inputs=network_inputs,
            sample_information=sample_information,
        )


def generate_model_samples_from_trial_info(model, trial_info, num_tasks: int, num_samples: int, device='cuda'):
    """Generate model samples given a fixed trial_info object."""
    model.eval()
    with torch.no_grad():
        if hasattr(trial_info, 'prep_network_inputs'):
            _, samples_dict = model.generate_samples(
                prep_network_inputs=trial_info.prep_network_inputs,
                diffusion_network_inputs=trial_info.diffusion_network_inputs,
                prep_epoch_durations=trial_info.prep_epoch_durations,
                diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
                samples_shape=[num_tasks, num_samples],
                noise_scaler=1.0,
            )
        else:
            samples_dict = model.generate_samples(
                network_input=trial_info.network_inputs.to(device),
                samples_shape=[num_tasks, num_samples],
                noise_scaler=1.0,
            )
    return samples_dict['samples']


def compute_pair_stats(a_samples, b_samples):
    """Compute mean/std shift between two sample sets."""
    a_mean = a_samples.mean(dim=0)
    b_mean = b_samples.mean(dim=0)
    mean_shift = torch.norm(a_mean - b_mean).item()

    a_std = a_samples.std(dim=0)
    b_std = b_samples.std(dim=0)
    std_change = torch.norm(a_std - b_std).item()

    return {'mean_shift': mean_shift, 'std_change': std_change}


def compare_per_task(real_samples, intact_samples, ablated_samples):
    """Compute per-task stats and average across tasks."""
    per_task_stats = []
    for task_idx in range(real_samples.shape[0]):
        real_task = real_samples[task_idx]
        intact_task = intact_samples[task_idx]
        ablated_task = ablated_samples[task_idx]

        stats = {
            'intact_vs_ablated': compute_pair_stats(intact_task, ablated_task),
            'real_vs_intact': compute_pair_stats(real_task, intact_task),
            'real_vs_ablated': compute_pair_stats(real_task, ablated_task),
        }
        per_task_stats.append(stats)

    avg_stats = {}
    for key in per_task_stats[0].keys():
        avg_stats[key] = {
            metric: float(np.mean([s[key][metric] for s in per_task_stats]))
            for metric in per_task_stats[0][key].keys()
        }
    return per_task_stats, avg_stats


def plot_per_task_samples(task, real_samples, intact_samples, ablated_samples, save_path):
    """Plot real/intact/ablated samples for each task instance."""
    num_tasks = real_samples.shape[0]
    is_2d = real_samples.shape[-1] == 2

    fig, axes = plt.subplots(num_tasks, 3, figsize=(9, 3 * num_tasks))
    if num_tasks == 1:
        axes = np.array([axes])

    for t in range(num_tasks):
        for col, (samples, title) in enumerate([
            (real_samples[t], 'Real'),
            (intact_samples[t], 'Intact'),
            (ablated_samples[t], 'Ablated'),
        ]):
            ax = axes[t, col]
            if is_2d:
                # Use the task's sample visualizer for color wheel styling
                task.sample_gen.display_samples(samples.unsqueeze(0), ax, batch_idx=0)
                if hasattr(task.sample_gen, "sample_radius"):
                    radius = float(task.sample_gen.sample_radius)
                    margin = radius * 0.2
                    ax.set_xlim(-radius - margin, radius + margin)
                    ax.set_ylim(-radius - margin, radius + margin)
                ax.set_aspect('equal')
            else:
                for dim in range(min(samples.shape[-1], 2)):
                    ax.hist(samples[:, dim].numpy(), bins=40, alpha=0.6, label=f'Dim {dim}')
                ax.legend()
            if t == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f'Task {t + 1}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_and_plot(intact_samples, ablated_samples, save_path='ablation_comparison.png'):
    """Compare and visualize intact vs ablated distributions."""
    print("\nComparing distributions...")
    
    # Compute statistics
    intact_mean = intact_samples.mean(dim=0)
    ablated_mean = ablated_samples.mean(dim=0)
    mean_shift = torch.norm(intact_mean - ablated_mean).item()
    
    intact_std = intact_samples.std(dim=0)
    ablated_std = ablated_samples.std(dim=0)
    std_change = torch.norm(intact_std - ablated_std).item()
    
    print(f"  Mean shift: {mean_shift:.4f}")
    print(f"  Std change: {std_change:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # If 2D, show scatter plots
    if intact_samples.shape[1] == 2:
        # Subsample for visibility
        n_plot = min(5000, len(intact_samples))
        
        axes[0].scatter(intact_samples[:n_plot, 0], intact_samples[:n_plot, 1], 
                       alpha=0.3, s=1, c='blue', label='Intact')
        axes[0].set_title('Intact Model')
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].set_aspect('equal')
        
        axes[1].scatter(ablated_samples[:n_plot, 0], ablated_samples[:n_plot, 1], 
                       alpha=0.3, s=1, c='red', label='Ablated')
        axes[1].set_title('Ablated Model')
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        axes[1].set_aspect('equal')
        
        # Overlay
        axes[2].scatter(intact_samples[:n_plot, 0], intact_samples[:n_plot, 1], 
                       alpha=0.2, s=1, c='blue', label='Intact')
        axes[2].scatter(ablated_samples[:n_plot, 0], ablated_samples[:n_plot, 1], 
                       alpha=0.2, s=1, c='red', label='Ablated')
        axes[2].set_title('Overlay')
        axes[2].legend()
        axes[2].set_aspect('equal')
    else:
        # For higher dimensions, show histograms
        for dim in range(min(intact_samples.shape[1], 2)):
            axes[0].hist(intact_samples[:, dim].numpy(), bins=50, alpha=0.7, label=f'Dim {dim}')
            axes[1].hist(ablated_samples[:, dim].numpy(), bins=50, alpha=0.7, label=f'Dim {dim}')
        axes[0].set_title('Intact Model')
        axes[1].set_title('Ablated Model')
        axes[0].legend()
        axes[1].legend()
        axes[2].text(0.5, 0.5, f'Mean shift: {mean_shift:.3f}\nStd change: {std_change:.3f}',
                    ha='center', va='center', fontsize=14)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {save_path}")
    plt.close()
    
    return {'mean_shift': mean_shift, 'std_change': std_change}


def main():
    """
    Main workflow demonstration.
    """
    print("=" * 70)
    print("NEURON ABLATION")
    print("=" * 70)
    
    # ===== Configuration =====
    RUN_PATH = "results_link_sampler/run_c3_index_cued_diffusion_0.3_swap_2"  # Change this for pro/retro run
    NEURONS_TO_ABLATE = list(range(16))  # 0-indexed
    NUM_TASKS = 8
    NUM_SAMPLES_PER_TASK = 512
    OUTPUT_DIR = "ablation_results"
    SAVE_PER_NEURON_RESULTS = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ===== Step 1: Load model =====
    print("\n[1/5] Loading trained model...")
    try:
        model, task, args = load_model_simple(RUN_PATH, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure to update RUN_PATH to point to your trained model!")
        return
    
    # ===== Step 2: Generate samples for fixed tasks =====
    print("\n[2/5] Generating samples from intact model...")
    task_variable_information = task.task_variable_gen.generate_variable_dict(
        batch_size=NUM_TASKS
    )
    trial_info = build_trial_information_for_fixed_tasks(
        task,
        task_variable_information,
        num_samples=NUM_SAMPLES_PER_TASK,
    )
    real_samples = trial_info.sample_information.sample_set.detach().cpu()
    intact_samples = generate_model_samples_from_trial_info(
        model,
        trial_info,
        num_tasks=NUM_TASKS,
        num_samples=NUM_SAMPLES_PER_TASK,
        device=device,
    ).detach().cpu()
    torch.save(real_samples, os.path.join(OUTPUT_DIR, 'real_samples.pt'))
    torch.save(intact_samples, os.path.join(OUTPUT_DIR, 'intact_samples.pt'))
    torch.save(task_variable_information, os.path.join(OUTPUT_DIR, 'task_variable_information.pt'))
    
    # Cache intact model weights so we can reset between ablations
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ===== Step 3-5: Loop over neurons =====
    import json
    all_neuron_metrics = {}

    for idx, neuron_idx in enumerate(NEURONS_TO_ABLATE, start=1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(NEURONS_TO_ABLATE)}] Ablating neuron {neuron_idx}")
        print("=" * 70)

        # Reset to intact model
        model.load_state_dict(base_state)

        # Ablate neuron
        ablate_neuron_simple(model, neuron_idx)

        # Generate ablated samples
        ablated_samples = generate_model_samples_from_trial_info(
            model,
            trial_info,
            num_tasks=NUM_TASKS,
            num_samples=NUM_SAMPLES_PER_TASK,
            device=device,
        ).detach().cpu()

        # Compare
        per_task_stats, avg_stats = compare_per_task(
            real_samples,
            intact_samples,
            ablated_samples,
        )

        if SAVE_PER_NEURON_RESULTS:
            neuron_dir = os.path.join(OUTPUT_DIR, f'neuron_{neuron_idx}')
            os.makedirs(neuron_dir, exist_ok=True)
            torch.save(
                ablated_samples,
                os.path.join(neuron_dir, 'ablated_samples.pt')
            )
            plot_per_task_samples(
                task,
                real_samples,
                intact_samples,
                ablated_samples,
                save_path=os.path.join(neuron_dir, 'comparison_per_task.png')
            )
            with open(os.path.join(neuron_dir, 'metrics.json'), 'w') as f:
                json.dump({
                    'neuron_ablated': neuron_idx,
                    'num_tasks': NUM_TASKS,
                    'num_samples_per_task': NUM_SAMPLES_PER_TASK,
                    'per_task_stats': per_task_stats,
                    'avg_stats': avg_stats,
                }, f, indent=2)

        all_neuron_metrics[neuron_idx] = {
            'per_task_stats': per_task_stats,
            'avg_stats': avg_stats,
        }

        print(f"Neuron {neuron_idx} average effects across tasks:")
        for key, stats in avg_stats.items():
            print(f"  {key}: mean_shift={stats['mean_shift']:.4f}, std_change={stats['std_change']:.4f}")

    # Save all metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics_all_neurons.json'), 'w') as f:
        json.dump({
            'num_tasks': NUM_TASKS,
            'num_samples_per_task': NUM_SAMPLES_PER_TASK,
            'neurons_tested': NEURONS_TO_ABLATE,
            'all_neuron_metrics': all_neuron_metrics,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("  - intact_samples.pt")
    print("  - real_samples.pt")
    print("  - task_variable_information.pt")
    print("  - metrics_all_neurons.json")
    if SAVE_PER_NEURON_RESULTS:
        print("  - neuron_*/ablated_samples.pt")
        print("  - neuron_*/comparison_per_task.png")
        print("  - neuron_*/metrics.json")
    

if __name__ == "__main__":
    main()
