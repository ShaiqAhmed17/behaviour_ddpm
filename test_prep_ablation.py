"""
Test that ablation is now applied during BOTH prep and diffusion phases.
"""

import torch
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart


def test_prep_ablation():
    """Test that prep states are now different when ablation is applied."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path('results_link_sampler/index_cued_first_diffusion_0.3_swap_7/args.yaml')
    
    print("="*70)
    print("TESTING PREP-PHASE ABLATION")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = model_path.parent / 'state.mdl'
    args, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(model_path), device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✓ Model loaded")
    
    # Prepare ablation vector (neuron 0)
    ablation_neuron = 0
    ablation_vector = model.behaviour_nullspace[ablation_neuron].clone()
    ablation_vector = ablation_vector / torch.norm(ablation_vector)
    ablation_vector = ablation_vector.to(device)
    print(f"\n✓ Prepared ablation vector for neuron {ablation_neuron}")
    
    # Generate a single trial
    trial = {'cue': 1, 'color1_angle': 45, 'color2_angle': 135, 'swap': False}
    
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
    
    task_variable_dict = task.task_variable_gen.generate_variable_dict(
        batch_size=1,
        override_stimulus_features_dict=override_stimulus_features_dict
    )
    
    trial_info = task.generate_trial_information(
        batch_size=1,
        num_samples=1,
        override_task_variable_information=task_variable_dict
    )
    
    # Generate WITHOUT ablation
    print("\n" + "-"*70)
    print("1. Generating WITHOUT ablation (healthy baseline)")
    print("-"*70)
    with torch.no_grad():
        prep_dicts_healthy, samples_dict_healthy = model.generate_samples(
            prep_network_inputs=trial_info.prep_network_inputs,
            diffusion_network_inputs=trial_info.diffusion_network_inputs,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[1, 1],
            ablation_vector=None,  # No ablation
        )
    
    # Extract prep states from all epochs
    prep_states_healthy = [pd['postprep_state'][0, 0].cpu().numpy() for pd in prep_dicts_healthy]
    final_sample_healthy = samples_dict_healthy['samples'][0, 0].cpu().numpy()
    
    print(f"Prep states (4 epochs): shapes = {[s.shape for s in prep_states_healthy]}")
    print(f"Final prep state L2 norm: {np.linalg.norm(prep_states_healthy[-1]):.2f}")
    print(f"Final prep state neuron {ablation_neuron} value: {prep_states_healthy[-1][ablation_neuron]:.4f}")
    print(f"Final sample: {final_sample_healthy}")
    
    # Generate WITH ablation
    print("\n" + "-"*70)
    print(f"2. Generating WITH ablation (neuron {ablation_neuron} ablated)")
    print("-"*70)
    with torch.no_grad():
        prep_dicts_ablated, samples_dict_ablated = model.generate_samples(
            prep_network_inputs=trial_info.prep_network_inputs,
            diffusion_network_inputs=trial_info.diffusion_network_inputs,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[1, 1],
            ablation_vector=ablation_vector,  # WITH ablation
        )
    
    prep_states_ablated = [pd['postprep_state'][0, 0].cpu().numpy() for pd in prep_dicts_ablated]
    final_sample_ablated = samples_dict_ablated['samples'][0, 0].cpu().numpy()
    
    print(f"Prep states (4 epochs): shapes = {[s.shape for s in prep_states_ablated]}")
    print(f"Final prep state L2 norm: {np.linalg.norm(prep_states_ablated[-1]):.2f}")
    print(f"Final prep state neuron {ablation_neuron} value: {prep_states_ablated[-1][ablation_neuron]:.4f}")
    print(f"Final sample: {final_sample_ablated}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON: Healthy vs Ablated")
    print("="*70)
    
    for epoch_idx in range(len(prep_states_healthy)):
        diff = np.linalg.norm(prep_states_healthy[epoch_idx] - prep_states_ablated[epoch_idx])
        print(f"\nEpoch {epoch_idx}:")
        print(f"  Prep state difference (L2): {diff:.4f}")
        print(f"  Healthy neuron {ablation_neuron}: {prep_states_healthy[epoch_idx][ablation_neuron]:.4f}")
        print(f"  Ablated neuron {ablation_neuron}: {prep_states_ablated[epoch_idx][ablation_neuron]:.4f}")
        
        # Check component along ablation direction
        component_healthy = np.dot(prep_states_healthy[epoch_idx], ablation_vector.cpu().numpy())
        component_ablated = np.dot(prep_states_ablated[epoch_idx], ablation_vector.cpu().numpy())
        print(f"  Component along ablation direction:")
        print(f"    Healthy: {component_healthy:.4f}")
        print(f"    Ablated: {component_ablated:.4f} ← Should be ~0!")
    
    sample_diff = np.linalg.norm(final_sample_healthy - final_sample_ablated)
    print(f"\nFinal sample difference (L2): {sample_diff:.4f}")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    final_component_ablated = np.dot(prep_states_ablated[-1], ablation_vector.cpu().numpy())
    final_diff = np.linalg.norm(prep_states_healthy[-1] - prep_states_ablated[-1])
    
    if abs(final_component_ablated) < 0.1 and final_diff > 1.0:
        print("✓ SUCCESS: Prep-phase ablation is working!")
        print(f"  - Ablated component ~0: {abs(final_component_ablated):.6f} < 0.1")
        print(f"  - Prep states differ: {final_diff:.2f} > 1.0")
        print(f"  - Behavior differs: {sample_diff:.2f}")
    else:
        print("✗ FAILURE: Prep-phase ablation may not be working correctly")
        print(f"  - Ablated component: {abs(final_component_ablated):.6f} (should be ~0)")
        print(f"  - Prep state difference: {final_diff:.2f} (should be > 1)")


if __name__ == '__main__':
    test_prep_ablation()
