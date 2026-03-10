"""
Analysis script for ablated teacher trajectories.

This script demonstrates how to:
1. Load saved trajectory data (from training or extraction)
2. Extract neural states at specific timepoints
3. Perform analyses similar to prospective_memory_analysis.ipynb

Usage:
    # From m-t-m training output:
    python analyze_ablated_teacher_trajectories.py <run_directory>
    
    # From extract_ablated_trajectories.py output:
    python analyze_ablated_teacher_trajectories.py <ablation_trajectories_dir> --extraction
    
Examples:
    python analyze_ablated_teacher_trajectories.py results/m-t-m_run_001/
    python analyze_ablated_teacher_trajectories.py results/model/ablation_trajectories_neuron_0/ --extraction
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_trajectory_data(run_dir, from_extraction=False):
    """
    Load the saved trajectory data.
    
    Args:
        run_dir: Directory containing trajectory data
        from_extraction: If True, load from extract_ablated_trajectories.py output
                        If False, load from m-t-m training output
    """
    if from_extraction:
        traj_path = os.path.join(run_dir, "ablated_trajectories.pt")
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"Trajectory file not found: {traj_path}\n"
                "Make sure you've run extract_ablated_trajectories.py first."
            )
        
        data = torch.load(traj_path, map_location='cpu', weights_only=False)
        print(f"Loaded extracted trajectory data")
        if 'metadata' in data:
            print(f"  Ablated neuron: {data['metadata']['ablate_neuron']}")
            print(f"  Number of trials: {data['metadata']['num_trials']}")
        
        # Normalize format to match training output structure
        normalized = {
            'ablated_teacher_trajectories': data.get('embedded_sample_trajectory'),  # Use full neural state
            'ablated_teacher_prep_dicts': data.get('prep_dicts', []),
            'trial_info': None,
            'metadata': data.get('metadata', {}),
        }
        # Also pass through other trajectory types
        if 'sample_trajectory' in data:
            normalized['sample_trajectory'] = data['sample_trajectory']
        if 'early_x0_preds' in data:
            normalized['early_x0_preds'] = data['early_x0_preds']
        if 'epsilon_hat' in data:
            normalized['epsilon_hat'] = data['epsilon_hat']
        
        return normalized
    else:
        traj_path = os.path.join(run_dir, "ablated_teacher_trajectories_latest.pt")
        
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"Trajectory file not found: {traj_path}\n"
                "Make sure the training run has completed at least one logging step."
            )
        
        data = torch.load(traj_path, map_location='cpu', weights_only=False)
        print(f"Loaded trajectory data from training step {data['training_step']}")
        
        return data


def extract_neural_states_at_timepoint(trajectories, timepoint_idx):
    """
    Extract neural states at a specific timepoint from trajectories.
    
    Args:
        trajectories: Tensor of shape [batch, num_samples, T, state_dim]
        timepoint_idx: Index of timepoint to extract (0 to T-1)
        
    Returns:
        Neural states of shape [batch, num_samples, state_dim]
    """
    if trajectories is None:
        raise ValueError("Trajectories not available in saved data")
    
    return trajectories[:, :, timepoint_idx, :]


def plot_trajectory_evolution(trajectories, trial_idx=0, save_path=None):
    """
    Plot the evolution of neural trajectories over time.
    
    Args:
        trajectories: Tensor of shape [batch, num_samples, T, state_dim]
        trial_idx: Which trial to plot (default: 0)
        save_path: Optional path to save the figure
    """
    if trajectories is None:
        print("No trajectories available to plot")
        return
    
    # Shape: [num_samples, T, state_dim]
    trial_trajs = trajectories[trial_idx].numpy()
    num_samples, T, state_dim = trial_trajs.shape
    
    # Plot first few dimensions over time
    dims_to_plot = min(6, state_dim)
    fig, axes = plt.subplots(dims_to_plot, 1, figsize=(12, 2*dims_to_plot), sharex=True)
    if dims_to_plot == 1:
        axes = [axes]
    
    for dim_idx in range(dims_to_plot):
        ax = axes[dim_idx]
        
        # Plot trajectories for first 10 samples
        for sample_idx in range(min(10, num_samples)):
            ax.plot(trial_trajs[sample_idx, :, dim_idx], alpha=0.5, linewidth=1)
        
        ax.set_ylabel(f'Dim {dim_idx}')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Diffusion Timestep')
    axes[0].set_title(f'Neural Trajectory Evolution (Trial {trial_idx})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_prep_trajectories(prep_dicts, trial_idx=0, save_path=None):
    """
    Plot preparatory epoch trajectories.
    
    Args:
        prep_dicts: List of preparatory dictionaries
        trial_idx: Which trial to plot
        save_path: Optional path to save the figure
    """
    num_epochs = len(prep_dicts)
    fig, axes = plt.subplots(num_epochs, 1, figsize=(12, 3*num_epochs), sharex=True)
    if num_epochs == 1:
        axes = [axes]
    
    for epoch_idx, prep_dict in enumerate(prep_dicts):
        ax = axes[epoch_idx]
        
        if 'preparatory_trajectory' in prep_dict:
            # Shape: [batch, num_samples, num_prep_steps, state_dim]
            traj = prep_dict['preparatory_trajectory'][trial_idx].numpy()
            num_samples, num_steps, state_dim = traj.shape
            
            # Plot first few dimensions
            dims_to_plot = min(3, state_dim)
            for dim_idx in range(dims_to_plot):
                for sample_idx in range(min(5, num_samples)):
                    ax.plot(traj[sample_idx, :, dim_idx], 
                           alpha=0.5, linewidth=1, 
                           label=f'Dim {dim_idx}' if sample_idx == 0 else '')
            
            ax.set_ylabel(f'Prep Epoch {epoch_idx}')
            ax.grid(True, alpha=0.3)
            if epoch_idx == 0:
                ax.legend()
    
    axes[-1].set_xlabel('Preparatory Steps')
    axes[0].set_title(f'Preparatory Trajectories (Trial {trial_idx})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prep trajectory plot to {save_path}")
    else:
        plt.show()
    
    return fig


def analyze_trajectory_statistics(trajectories, name="trajectories"):
    """
    Compute basic statistics about the trajectories.
    
    Args:
        trajectories: Tensor of shape [batch, num_samples, T, state_dim]
        name: Name of the trajectory type for display
    """
    if trajectories is None:
        print(f"\n=== {name.title()} Statistics ===")
        print("Not available in this data")
        return
    
    print(f"\n=== {name.title()} Statistics ===")
    print(f"Shape: {trajectories.shape}")
    print(f"  batch_size: {trajectories.shape[0]}")
    print(f"  num_samples: {trajectories.shape[1]}")
    print(f"  num_timesteps: {trajectories.shape[2]}")
    print(f"  dims: {trajectories.shape[3]}")
    print(f"\nValue range:")
    print(f"  min: {trajectories.min().item():.4f}")
    print(f"  max: {trajectories.max().item():.4f}")
    print(f"  mean: {trajectories.mean().item():.4f}")
    print(f"  std: {trajectories.std().item():.4f}")
    
    # Compute trajectory norms over time
    traj_norms = torch.norm(trajectories, dim=-1).mean(dim=(0, 1))  # [T]
    print(f"\nMean trajectory norm:")
    print(f"  at t=0: {traj_norms[0].item():.4f}")
    print(f"  at t=T/2: {traj_norms[len(traj_norms)//2].item():.4f}")
    print(f"  at t=T: {traj_norms[-1].item():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablated teacher trajectories"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Directory containing trajectory data"
    )
    parser.add_argument(
        "--extraction",
        action="store_true",
        help="Load from extract_ablated_trajectories.py output (not training output)"
    )
    parser.add_argument(
        "--trial-idx",
        type=int,
        default=0,
        help="Trial index to visualize (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: run_dir/trajectory_analysis)"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.run_dir, "trajectory_analysis")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print(f"\nLoading data from {args.run_dir}...")
    data = load_trajectory_data(args.run_dir, from_extraction=args.extraction)
    
    # Get trajectories - try different sources in order of preference
    # 1. ablated_teacher_trajectories (from m-t-m training, if neural states were saved)
    # 2. embedded_sample_trajectory (full neural state from extraction)
    # 3. sample_trajectory (behavioral subspace only)
    trajectories = data.get('ablated_teacher_trajectories')
    if trajectories is None:
        trajectories = data.get('embedded_sample_trajectory')
    
    prep_dicts = data.get('ablated_teacher_prep_dicts', [])
    
    # Check for alternative trajectory data
    sample_trajectory = data.get('sample_trajectory') if 'sample_trajectory' in data else None
    early_x0_preds = data.get('early_x0_preds') if 'early_x0_preds' in data else None
    epsilon_hat = data.get('epsilon_hat') if 'epsilon_hat' in data else None
    
    # Determine what we have
    if trajectories is not None:
        traj_name = "Full Neural State Trajectories"
        primary_traj = trajectories
    elif sample_trajectory is not None:
        traj_name = "Behavioral Subspace Trajectories"
        primary_traj = sample_trajectory
    else:
        traj_name = None
        primary_traj = None
    
    # Analyze statistics for all available trajectory types
    if primary_traj is not None:
        analyze_trajectory_statistics(primary_traj, name=traj_name)
    if sample_trajectory is not None and trajectories is not None:
        analyze_trajectory_statistics(sample_trajectory, name="Behavioral Subspace Trajectories")
    if early_x0_preds is not None:
        analyze_trajectory_statistics(early_x0_preds, name="Early X0 Predictions")
    if epsilon_hat is not None:
        analyze_trajectory_statistics(epsilon_hat, name="Predicted Residuals (Epsilon Hat)")
    
    # Plot diffusion trajectories
    if primary_traj is not None:
        print(f"\nPlotting {traj_name.lower()} for trial {args.trial_idx}...")
        plot_trajectory_evolution(
            primary_traj,
            trial_idx=args.trial_idx,
            save_path=os.path.join(args.output_dir, f"diffusion_trajectories_trial_{args.trial_idx}.png")
        )
    
    # Plot preparatory trajectories
    if prep_dicts and len(prep_dicts) > 0:
        print(f"\nPlotting preparatory trajectories for trial {args.trial_idx}...")
        plot_prep_trajectories(
            prep_dicts,
            trial_idx=args.trial_idx,
            save_path=os.path.join(args.output_dir, f"prep_trajectories_trial_{args.trial_idx}.png")
        )
    
    print(f"\n✓ Analysis complete! Results saved to {args.output_dir}")
    
    # Example: Extract states at specific timepoint
    if primary_traj is not None:
        print(f"\n=== Example: Extracting states at timepoint T/2 ===")
        mid_timepoint = primary_traj.shape[2] // 2
        states_at_mid = extract_neural_states_at_timepoint(primary_traj, mid_timepoint)
        print(f"States at timepoint {mid_timepoint}:")
        print(f"  Shape: {states_at_mid.shape}")
        print(f"  Mean: {states_at_mid.mean().item():.4f}")
        print(f"  Std: {states_at_mid.std().item():.4f}")


if __name__ == "__main__":
    main()
