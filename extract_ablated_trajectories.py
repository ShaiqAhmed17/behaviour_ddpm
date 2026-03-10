"""
Extract trajectories from an ablated teacher model without retraining.

This script loads a trained model, applies ablation to a specified neuron,
generates samples, and saves the trajectories for downstream analysis.

Usage:
    python extract_ablated_trajectories.py \
        --source_run_path <path_to_model> \
        --ablate_neuron <neuron_index> \
        --num_trials <number_of_trials> \
        --output_dir <output_directory>
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from purias_utils.util.arguments_yaml import ConfigNamepace
from ddpm import tasks, model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract trajectories from ablated model"
    )
    parser.add_argument(
        "--source_run_path",
        type=str,
        required=True,
        help="Run directory with args.yaml + checkpoint",
    )
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default="state.mdl",
        help="Checkpoint filename (default: state.mdl)",
    )
    parser.add_argument(
        "--ablate_neuron",
        type=int,
        required=True,
        help="Neuron index to ablate (indexing the memory nullspace)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of trials to generate (default: 100)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples per trial (default: 500)",
    )
    parser.add_argument(
        "--noise_scaler",
        type=float,
        default=None,
        help="Noise scaler for sampling (default: use model's trained setting)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: source_run_path/ablation_trajectories_neuron_N)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for trial generation (default: same as num_trials)",
    )
    return parser.parse_args()


def load_model_from_run(run_path, device='cuda', checkpoint_name='state.mdl'):
    """Load a trained model from a run directory."""
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

    print(f"✓ Loaded model from {run_path}")
    return ddpm_model, task, args


def get_ablation_vector(model_instance, neuron_idx):
    """Extract the ablation vector from the model's nullspace."""
    print(f"\n[Ablation] Extracting nullspace direction {neuron_idx}")
    print(f"  behaviour_nullspace shape: {model_instance.behaviour_nullspace.shape}")
    
    ablation_vector = model_instance.behaviour_nullspace[neuron_idx].clone()
    ablation_vector = ablation_vector / torch.norm(ablation_vector)
    
    print(f"  ablation_vector shape: {ablation_vector.shape}")
    print(f"  normalized norm: {torch.norm(ablation_vector).item():.6f}")
    
    return ablation_vector


def move_inputs_to_device(inputs, device):
    """Move network inputs to specified device."""
    if isinstance(inputs, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
    else:
        return [v.to(device) if isinstance(v, torch.Tensor) else v 
                for v in inputs]


def extract_trajectories_batch(
    model, 
    task, 
    ablation_vector, 
    batch_size, 
    num_samples,
    noise_scaler=None,
    device='cuda'
):
    """Extract trajectories for a single batch of trials."""
    
    # Generate trial information
    trial_info = task.generate_trial_information(
        batch_size=batch_size, 
        num_samples=num_samples
    )
    
    with torch.no_grad():
        # Prepare inputs
        prep_inputs = move_inputs_to_device(trial_info.prep_network_inputs, device)
        diff_inputs = move_inputs_to_device(trial_info.diffusion_network_inputs, device)
        
        # Build kwargs for sampling with ablation
        sample_kwargs = {
            'prep_network_inputs': prep_inputs,
            'diffusion_network_inputs': diff_inputs,
            'prep_epoch_durations': trial_info.prep_epoch_durations,
            'diffusion_epoch_durations': trial_info.diffusion_epoch_durations,
            'samples_shape': [batch_size, num_samples],
            'ablation_vector': ablation_vector,
        }
        if noise_scaler is not None:
            sample_kwargs['noise_scaler'] = noise_scaler
        
        # Generate samples with trajectories
        prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)
    
    # Extract all relevant data
    # Note: 'embedded_sample_trajectory' contains full neural state trajectories
    # while 'sample_trajectory' is projected to behavioral subspace
    result = {
        'samples': samples_dict['samples'].detach().cpu(),
        'embedded_sample_trajectory': samples_dict.get('embedded_sample_trajectory', None),  # Full neural state
        'sample_trajectory': samples_dict.get('sample_trajectory', None),  # Behavioral subspace
        'early_x0_preds': samples_dict.get('early_x0_preds', None),
        'epsilon_hat': samples_dict.get('epsilon_hat', None),
        'prep_dicts': [{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                       for k, v in d.items()} for d in prep_dicts],
        'trial_info': trial_info,
    }
    
    # Move tensors to CPU
    for key in ['embedded_sample_trajectory', 'sample_trajectory', 'early_x0_preds', 'epsilon_hat']:
        if result[key] is not None:
            result[key] = result[key].detach().cpu()
    
    return result


def main():
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model and task
    print(f"\nLoading model from {args.source_run_path}...")
    model_instance, task, model_args = load_model_from_run(
        args.source_run_path,
        device=device,
        checkpoint_name=args.source_checkpoint,
    )
    
    # Validate and extract ablation vector
    memory_dims = model_instance.behaviour_nullspace.shape[0]
    if args.ablate_neuron < 0 or args.ablate_neuron >= memory_dims:
        raise ValueError(
            f"Neuron index {args.ablate_neuron} out of range for memory subspace (0-{memory_dims-1})"
        )
    
    ablation_vector = get_ablation_vector(model_instance, args.ablate_neuron)
    ablation_vector = ablation_vector.to(device)
    
    print(f"\nModel has {memory_dims} nullspace basis vectors")
    print(f"Ablating nullspace direction {args.ablate_neuron}")
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.source_run_path, 
            f"ablation_trajectories_neuron_{args.ablate_neuron}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Determine batch size
    batch_size = args.batch_size if args.batch_size is not None else args.num_trials
    num_batches = (args.num_trials + batch_size - 1) // batch_size
    
    print(f"\nGenerating {args.num_trials} trials in {num_batches} batch(es)...")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples per trial: {args.num_samples}")
    if args.noise_scaler is not None:
        print(f"  Noise scaler: {args.noise_scaler}")
    
    # Extract trajectories
    all_results = []
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting trajectories"):
        # Adjust batch size for last batch if needed
        current_batch_size = min(batch_size, args.num_trials - batch_idx * batch_size)
        
        result = extract_trajectories_batch(
            model_instance,
            task,
            ablation_vector,
            current_batch_size,
            args.num_samples,
            noise_scaler=args.noise_scaler,
            device=device
        )
        
        all_results.append(result)
    
    # Concatenate results
    print("\nConcatenating results...")
    combined_result = {
        'samples': torch.cat([r['samples'] for r in all_results], dim=0),
        'embedded_sample_trajectory': torch.cat([r['embedded_sample_trajectory'] for r in all_results], dim=0) if all_results[0]['embedded_sample_trajectory'] is not None else None,
        'sample_trajectory': torch.cat([r['sample_trajectory'] for r in all_results], dim=0) if all_results[0]['sample_trajectory'] is not None else None,
        'early_x0_preds': torch.cat([r['early_x0_preds'] for r in all_results], dim=0) if all_results[0]['early_x0_preds'] is not None else None,
        'epsilon_hat': torch.cat([r['epsilon_hat'] for r in all_results], dim=0) if all_results[0]['epsilon_hat'] is not None else None,
        'prep_dicts': all_results[0]['prep_dicts'],  # Just keep first batch's prep structure
        'metadata': {
            'source_run_path': args.source_run_path,
            'source_checkpoint': args.source_checkpoint,
            'ablate_neuron': args.ablate_neuron,
            'num_trials': args.num_trials,
            'num_samples': args.num_samples,
            'noise_scaler': args.noise_scaler,
            'memory_dims': memory_dims,
        }
    }
    
    # Print statistics
    print("\n=== Extraction Summary ===")
    print(f"Samples shape: {combined_result['samples'].shape}")
    if combined_result['embedded_sample_trajectory'] is not None:
        print(f"Embedded sample trajectory shape: {combined_result['embedded_sample_trajectory'].shape}")
        print(f"  [batch, num_samples, T, full_neural_dim]")
    if combined_result['sample_trajectory'] is not None:
        print(f"Sample trajectory shape: {combined_result['sample_trajectory'].shape}")
        print(f"  [batch, num_samples, T, behavioral_dim]")
    if combined_result['early_x0_preds'] is not None:
        print(f"Early x0 preds shape: {combined_result['early_x0_preds'].shape}")
    if combined_result['epsilon_hat'] is not None:
        print(f"Epsilon hat shape: {combined_result['epsilon_hat'].shape}")
    
    # Save results
    output_path = os.path.join(args.output_dir, "ablated_trajectories.pt")
    print(f"\nSaving to {output_path}...")
    torch.save(combined_result, output_path)
    
    # Save metadata as JSON for easy inspection
    import json
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(combined_result['metadata'], f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_path}")
    print(f"✓ Extraction complete!")
    
    # Quick visualization
    if combined_result['embedded_sample_trajectory'] is not None:
        print("\n=== Quick Statistics (Full Neural State Trajectories) ===")
        trajs = combined_result['embedded_sample_trajectory']
        print(f"Trajectory value range: [{trajs.min():.4f}, {trajs.max():.4f}]")
        print(f"Trajectory mean: {trajs.mean():.4f}")
        print(f"Trajectory std: {trajs.std():.4f}")
    elif combined_result['sample_trajectory'] is not None:
        print("\n=== Quick Statistics (Behavioral Subspace Trajectories) ===")
        trajs = combined_result['sample_trajectory']
        print(f"Trajectory value range: [{trajs.min():.4f}, {trajs.max():.4f}]")
        print(f"Trajectory mean: {trajs.mean():.4f}")
        print(f"Trajectory std: {trajs.std():.4f}")


if __name__ == "__main__":
    main()
