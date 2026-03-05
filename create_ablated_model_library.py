"""
Create a library of ablated model variants from a healthy source model.

This script:
1. Loads a trained healthy model (e.g., index_cued_first_diffusion_0.3_swap_7)
2. Extracts all memory nullspace directions (e.g., 14 dimensions)
3. Creates a directory for each ablated variant with:
   - args.yaml (copy of original config)
   - state.mdl (copy of original checkpoint)
   - ablation_data.pt (ablation vector and metadata)
4. Tests each ablated variant to verify it generates samples correctly

Usage:
    python create_ablated_model_library.py \
        --source_run_path results_link_sampler/index_cued_first_diffusion_0.3_swap_7 \
        --output_base results_link_sampler/ablated_index_cued_models \
        --device cuda
"""

import argparse
import os
import shutil
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from purias_utils.util.arguments_yaml import ConfigNamepace
from ddpm import tasks, model


def load_model_from_run(run_path, device='cuda', checkpoint_name='state.mdl'):
    """Load a trained DDPM model from a run directory."""
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

    return ddpm_model, task, args


def get_ablation_vector(model_instance, neuron_idx):
    """
    Extract the ablation vector from the model's nullspace.
    
    neuron_idx: index in the memory nullspace (0 to num_memory_neurons-1)
    
    Returns: normalized nullspace direction vector (shape: [ambient_dim])
    """
    ablation_vector = model_instance.behaviour_nullspace[neuron_idx].clone()
    
    # Should already be normalized from SVD, but ensure it
    ablation_vector = ablation_vector / torch.norm(ablation_vector)
    
    return ablation_vector


def test_ablated_model(model_instance, task, ablation_vector, device, batch_size=5, num_samples=100):
    """Generate test samples from an ablated model to verify it works."""
    with torch.no_grad():
        trial_information = task.generate_trial_information(
            batch_size=batch_size, 
            num_samples=num_samples
        )
        
        # Move inputs to device
        if isinstance(trial_information.prep_network_inputs, dict):
            prep_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in trial_information.prep_network_inputs.items()}
        else:
            prep_inputs = [v.to(device) if isinstance(v, torch.Tensor) else v 
                          for v in trial_information.prep_network_inputs]
        
        if isinstance(trial_information.diffusion_network_inputs, dict):
            diff_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in trial_information.diffusion_network_inputs.items()}
        else:
            diff_inputs = [v.to(device) if isinstance(v, torch.Tensor) else v 
                          for v in trial_information.diffusion_network_inputs]
        
        # Generate ablated samples
        sample_kwargs = {
            'prep_network_inputs': prep_inputs,
            'diffusion_network_inputs': diff_inputs,
            'prep_epoch_durations': trial_information.prep_epoch_durations,
            'diffusion_epoch_durations': trial_information.diffusion_epoch_durations,
            'samples_shape': [batch_size, num_samples],
            'ablation_vector': ablation_vector.to(device),
        }
        
        prep_dicts, sample_dict = model_instance.generate_samples(**sample_kwargs)
        samples = sample_dict['samples']
        
        return samples, prep_dicts, sample_dict


def create_ablated_model_library(source_run_path, output_base, device='cuda', 
                                  checkpoint_name='state.mdl', test_samples=True):
    """
    Create a library of ablated model variants.
    
    Args:
        source_run_path: Path to healthy source model directory
        output_base: Base directory for ablated model library
        device: Device to use for loading/testing
        checkpoint_name: Name of checkpoint file in source directory
        test_samples: Whether to test each ablated variant
    """
    print(f"Loading source model from: {source_run_path}")
    source_model, task, args = load_model_from_run(source_run_path, device, checkpoint_name)
    
    # Get number of memory nullspace dimensions
    num_memory_dims = source_model.behaviour_nullspace.shape[0]
    ambient_dim = source_model.behaviour_nullspace.shape[1]
    
    print(f"\n{'='*70}")
    print(f"Source model information:")
    print(f"  Memory nullspace dimensions: {num_memory_dims}")
    print(f"  Ambient dimension: {ambient_dim}")
    print(f"  Creating {num_memory_dims} ablated variants")
    print(f"{'='*70}\n")
    
    # Create output base directory
    os.makedirs(output_base, exist_ok=True)
    
    # Save library metadata
    library_metadata = {
        'source_run_path': source_run_path,
        'source_checkpoint': checkpoint_name,
        'num_ablations': num_memory_dims,
        'ambient_dim': ambient_dim,
        'behavioral_dim': source_model.sample_subspace_accessor.shape[0],
        'created_date': str(np.datetime64('now')),
    }
    
    with open(os.path.join(output_base, 'library_metadata.json'), 'w') as f:
        json.dump(library_metadata, f, indent=2)
    
    print(f"Saved library metadata to: {os.path.join(output_base, 'library_metadata.json')}")
    
    # Create ablated variants
    successful_ablations = []
    failed_ablations = []
    
    for neuron_idx in tqdm(range(num_memory_dims), desc="Creating ablated variants"):
        try:
            # Create directory for this ablation
            ablation_dir = os.path.join(output_base, f'ablation_neuron_{neuron_idx}')
            os.makedirs(ablation_dir, exist_ok=True)
            
            # Extract ablation vector
            ablation_vector = get_ablation_vector(source_model, neuron_idx)
            
            # Copy source files
            source_args_path = os.path.join(source_run_path, 'args.yaml')
            source_state_path = os.path.join(source_run_path, checkpoint_name)
            
            shutil.copy(source_args_path, os.path.join(ablation_dir, 'args.yaml'))
            shutil.copy(source_state_path, os.path.join(ablation_dir, 'state.mdl'))
            
            # Save ablation data
            ablation_data = {
                'ablation_vector': ablation_vector.cpu(),
                'neuron_idx': neuron_idx,
                'behaviour_nullspace': source_model.behaviour_nullspace.detach().cpu(),
                'sample_subspace_accessor': source_model.sample_subspace_accessor.detach().cpu(),
                'behaviour_nullspace_accessor': source_model.behaviour_nullspace_accessor.detach().cpu(),
                'auxiliary_embedding_matrix': source_model.auxiliary_embedding_matrix.detach().cpu(),
            }
            torch.save(ablation_data, os.path.join(ablation_dir, 'ablation_data.pt'))
            
            # Save individual metadata
            ablation_metadata = {
                'ablation_neuron_idx': neuron_idx,
                'source_run_path': source_run_path,
                'source_checkpoint': checkpoint_name,
                'ablation_vector_norm': torch.norm(ablation_vector).item(),
            }
            
            with open(os.path.join(ablation_dir, 'ablation_metadata.json'), 'w') as f:
                json.dump(ablation_metadata, f, indent=2)
            
            # Test ablated model if requested
            if test_samples:
                samples, _, _ = test_ablated_model(
                    source_model, task, ablation_vector, device,
                    batch_size=3, num_samples=50
                )
                
                # Verify samples are valid (not NaN, reasonable range)
                if torch.isnan(samples).any():
                    raise ValueError(f"Generated samples contain NaN")
                if torch.isinf(samples).any():
                    raise ValueError(f"Generated samples contain Inf")
                
                # Save test sample statistics
                test_stats = {
                    'mean': samples.mean().item(),
                    'std': samples.std().item(),
                    'min': samples.min().item(),
                    'max': samples.max().item(),
                    'shape': list(samples.shape),
                }
                
                with open(os.path.join(ablation_dir, 'test_stats.json'), 'w') as f:
                    json.dump(test_stats, f, indent=2)
            
            successful_ablations.append(neuron_idx)
            
        except Exception as e:
            print(f"\n✗ Failed to create ablation variant {neuron_idx}: {e}")
            failed_ablations.append((neuron_idx, str(e)))
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successfully created: {len(successful_ablations)}/{num_memory_dims} ablated variants")
    
    if failed_ablations:
        print(f"✗ Failed: {len(failed_ablations)} variants")
        for idx, error in failed_ablations:
            print(f"  - Neuron {idx}: {error}")
    
    print(f"\nOutput directory: {output_base}")
    print(f"\nDirectory structure:")
    print(f"  {output_base}/")
    print(f"    ├── library_metadata.json")
    print(f"    ├── ablation_neuron_0/")
    print(f"    │   ├── args.yaml")
    print(f"    │   ├── state.mdl")
    print(f"    │   ├── ablation_data.pt")
    print(f"    │   ├── ablation_metadata.json")
    print(f"    │   └── test_stats.json")
    print(f"    ├── ablation_neuron_1/")
    print(f"    │   └── ...")
    print(f"    └── ...")
    print(f"\nYou can now analyze each ablated model using prospective_memory_analysis.ipynb")
    print(f"by pointing ARGS_PATH and CHECKPOINT_PATH to each directory.")
    print(f"{'='*70}\n")
    
    return successful_ablations, failed_ablations


def main():
    parser = argparse.ArgumentParser(
        description="Create library of ablated model variants"
    )
    parser.add_argument(
        "--source_run_path",
        type=str,
        required=True,
        help="Path to source healthy model directory"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=None,
        help="Base directory for ablated model library (default: source_path/../ablated_SOURCE_NAME)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="state.mdl",
        help="Name of checkpoint file in source directory"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip testing sample generation for each variant"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_base is None:
        source_name = Path(args.source_run_path).name
        parent_dir = Path(args.source_run_path).parent
        args.output_base = str(parent_dir / f"ablated_{source_name}")
    
    # Create library
    successful, failed = create_ablated_model_library(
        source_run_path=args.source_run_path,
        output_base=args.output_base,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        test_samples=not args.skip_test
    )
    
    return successful, failed


if __name__ == "__main__":
    main()
