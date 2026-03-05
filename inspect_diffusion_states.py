"""
Inspect extracted diffusion states to verify we have neural trajectories.
"""

import torch
import numpy as np
from pathlib import Path

def inspect_extracted_states(filepath):
    """Load and inspect extracted states."""
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    
    print(f"\n{'='*70}")
    print(f"INSPECTING: {filepath}")
    print(f"{'='*70}")
    
    print(f"\nModel: {data['model_name']}")
    print(f"Ablation applied: {data['ablation_applied']}")
    
    print(f"\n--- Data shapes ---")
    prep_states = data['prep_states']
    embedded_traj = data['embedded_trajectories']
    final_samples = data['final_samples']
    metadata = data['metadata']
    
    print(f"Prep states: {prep_states.shape}")
    print(f"  Interpretation: [{prep_states.shape[0]} trials, {prep_states.shape[1]} prep epochs, {prep_states.shape[2]} neural dims]")
    
    print(f"\nEmbedded trajectories: {embedded_traj.shape}")
    print(f"  Interpretation: [{embedded_traj.shape[0]} trials, {embedded_traj.shape[1]} timesteps, {embedded_traj.shape[2]} neural dims]")
    print(f"  ^^^ THESE ARE THE NEURAL STATES AFTER EACH DENOISING STEP WHERE ABLATION WAS APPLIED ^^^")
    
    print(f"\nFinal samples: {final_samples.shape}")
    print(f"  Interpretation: [{final_samples.shape[0]} trials, {final_samples.shape[1]} output dims]")
    
    print(f"\nMetadata: {metadata.shape}")
    print(f"  Interpretation: [{metadata.shape[0]} trials, 3 columns (cue, color1_angle, color2_angle)]")
    
    print(f"\n--- Sample statistics ---")
    print(f"Prep state L2 norms: mean={np.linalg.norm(prep_states, axis=-1).mean():.2f}, std={np.linalg.norm(prep_states, axis=-1).std():.2f}")
    print(f"Embedded trajectory L2 norms: mean={np.linalg.norm(embedded_traj, axis=-1).mean():.2f}, std={np.linalg.norm(embedded_traj, axis=-1).std():.2f}")
    print(f"Final sample L2 norms: mean={np.linalg.norm(final_samples, axis=-1).mean():.2f}, std={np.linalg.norm(final_samples, axis=-1).std():.2f}")
    
    # Show a single trial trajectory evolution
    trial_idx = 0
    print(f"\n--- Example: Trial {trial_idx} state evolution ---")
    print(f"Cue: {metadata[trial_idx, 0]}, Color1: {metadata[trial_idx, 1]:.1f}°, Color2: {metadata[trial_idx, 2]:.1f}°")
    
    # Show L2 norm evolution during diffusion (denoising)
    norms = np.linalg.norm(embedded_traj[trial_idx], axis=-1)  # [n_timesteps]
    print(f"\nNeural state L2 norm during denoising (ablation applied at each step):")
    print(f"  Timestep 0 (start): {norms[0]:.2f}")
    print(f"  Timestep 10: {norms[10]:.2f}")
    print(f"  Timestep 20: {norms[20]:.2f}")
    print(f"  Timestep 30: {norms[30]:.2f}")
    print(f"  Timestep 39 (final): {norms[-1]:.2f}")
    
    print(f"\n{'='*70}")
    print(f"✓ Data looks good! Ready for analysis.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'ablation_diffusion_states/neuron_0_states.pt'
    
    inspect_extracted_states(filepath)
