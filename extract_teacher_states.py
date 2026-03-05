"""
This script extracts neural states from ablated teachers at prep_idx=2 (delay 1 onset)
just like the prospective_memory_analysis.ipynb notebook does, but for teachers.

The extracted states can then be visualized using the same PCA/2D binning code in the notebook.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart


def generate_trial_combinations(angle_step=30):
    """Generate all non-swap trial combinations."""
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for color1 in angles:
            for color2 in angles:
                trials.append({
                    'cue': cue,
                    'color1_angle': color1,
                    'color2_angle': color2,
                    'swap': False
                })
    return trials


def extract_neural_state_from_model(trial, task, model, device, prep_idx=2, ablation_vector=None):
    """
    Extract 16D neural state for a single trial at specified timepoint.
    Same as in prospective_memory_analysis.ipynb
    """
    with torch.no_grad():
        # Convert angles to radians
        probe_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180)
        report_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180)
        
        override_stimulus_features = {
            'probe_features': probe_features,
            'report_features': report_features,
        }
        override_stimulus_cart_features = {}
        for k in override_stimulus_features.keys():
            override_stimulus_cart_features[f'{k}_cart'] = torch.stack(
                polar2cart(1.0, override_stimulus_features[k]), -1
            )
        
        override_stimulus_features_dict = {}
        for k, v in override_stimulus_features.items():
            override_stimulus_features_dict[k] = v
        for k, v in override_stimulus_cart_features.items():
            override_stimulus_features_dict[k] = v
        
        override_stimulus_features_dict['cued_item_idx'] = torch.tensor([trial['cue'] - 1])
        
        # Generate trial information
        task_variable_dict = task.task_variable_gen.generate_variable_dict(
            batch_size=1,
            override_stimulus_features_dict=override_stimulus_features_dict
        )
        
        trial_info = task.generate_trial_information(
            batch_size=1,
            num_samples=1,
            override_task_variable_information=task_variable_dict
        )
        
        # Move to device
        prep_network_inputs_device = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in trial_info.prep_network_inputs
        ]
        
        diffusion_network_inputs_device = [
            inp.to(device) if isinstance(inp, torch.Tensor) else inp
            for inp in trial_info.diffusion_network_inputs
        ]
        
        # Build sampling kwargs
        sample_kwargs = {
            'prep_network_inputs': prep_network_inputs_device,
            'diffusion_network_inputs': diffusion_network_inputs_device,
            'prep_epoch_durations': trial_info.prep_epoch_durations,
            'diffusion_epoch_durations': trial_info.diffusion_epoch_durations,
            'samples_shape': [1, 1],
            'noise_scaler': 1.0,
        }
        
        # Add ablation vector if provided
        if ablation_vector is not None:
            sample_kwargs['ablation_vector'] = ablation_vector
        
        # Run model
        prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)
        
        # Extract neural state
        neural_state = prep_dicts[prep_idx]['postprep_state'][0, 0, :16].cpu().numpy()
        
        return neural_state


def extract_teacher_states(teacher_path, prep_idx=2, n_trials=None):
    """
    Extract neural states from a teacher model (healthy or ablated).
    
    Args:
        teacher_path: Path to teacher args.yaml
        prep_idx: Which prep epoch to extract (default 2 = delay 1 onset)
        n_trials: Number of trials (None = all)
        
    Returns:
        Dictionary with neural_states, metadata, and model info
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_path = Path(teacher_path)
    
    print(f"{'='*70}")
    print(f"EXTRACTING TEACHER NEURAL STATES")
    print(f"{'='*70}")
    print(f"Teacher: {teacher_path.parent.name}")
    print(f"Prep index: {prep_idx}")
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model...")
    checkpoint_path = teacher_path.parent / 'state.mdl'
    args, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(teacher_path), device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✓ Model loaded")
    
    # Check for ablation
    ablation_vector = None
    ablation_neuron = None
    ablation_data_path = teacher_path.parent / 'ablation_data.pt'
    
    if ablation_data_path.exists():
        print(f"\n⚠️  ABLATION DETECTED")
        ablation_data = torch.load(ablation_data_path, map_location=device, weights_only=True)
        ablation_vector = ablation_data['ablation_vector'].to(device)
        ablation_neuron = ablation_data['neuron_idx']
        print(f"  Ablation neuron: {ablation_neuron}")
        print(f"  Ablation vector norm: {torch.norm(ablation_vector).item():.6f}")
        print(f"  ✓ Ablation will be applied during extraction")
    else:
        print(f"\n✓ No ablation - extracting healthy teacher states")
    
    # Generate trials
    trials = generate_trial_combinations()
    if n_trials is not None:
        trials = trials[:n_trials]
    
    print(f"\nExtracting neural states...")
    print(f"  Total trials: {len(trials)}")
    
    # Extract states
    neural_states = []
    metadata = []
    
    for trial in tqdm(trials, desc="Processing trials"):
        state = extract_neural_state_from_model(
            trial, task, model, device, 
            prep_idx=prep_idx, 
            ablation_vector=ablation_vector
        )
        neural_states.append(state)
        metadata.append([trial['cue'], trial['color1_angle'], trial['color2_angle']])
    
    neural_states = np.array(neural_states)
    metadata = np.array(metadata)
    
    print(f"\n✓ Extracted {len(neural_states)} neural states")
    print(f"  Shape: {neural_states.shape}")
    print(f"  Mean: {neural_states.mean():.4f}, Std: {neural_states.std():.4f}")
    
    return {
        'neural_states': neural_states,
        'metadata': metadata,
        'teacher_name': teacher_path.parent.name,
        'ablation_neuron': ablation_neuron,
        'prep_idx': prep_idx,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract teacher neural states')
    parser.add_argument('--teacher_path', type=str, required=True,
                       help='Path to teacher args.yaml')
    parser.add_argument('--prep_idx', type=int, default=2,
                       help='Prep epoch to extract (default: 2 = delay 1)')
    parser.add_argument('--n_trials', type=int, default=None,
                       help='Number of trials (default: all)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Where to save extracted states')
    
    args = parser.parse_args()
    
    # Extract states
    data = extract_teacher_states(
        args.teacher_path,
        prep_idx=args.prep_idx,
        n_trials=args.n_trials
    )
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        neural_states=data['neural_states'],
        metadata=data['metadata'],
        teacher_name=data['teacher_name'],
        ablation_neuron=data['ablation_neuron'],
        prep_idx=data['prep_idx']
    )
    
    print(f"\n{'='*70}")
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*70}")
