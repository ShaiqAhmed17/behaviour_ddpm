"""Shared 3D trajectory plotting functions for prospective memory analysis.

This module contains reusable plotting and data processing functions for
visualizing neural trajectories across prep and diffusion timesteps in a
shared PCA space.

Usage in notebook:
    from prospective_memory_plotting import (
        make_full_step_trajectory_dataset,
        make_full_step_diffusion_dataset,
        plot_global_3d_trajectories,
    )
"""

import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from ddpm.utils.vis.style import set_publication_style, save_figure, save_legend


set_publication_style()


def bin_angle(angle, bin_size):
    """Bin an angle into discrete buckets."""
    return int(angle // bin_size) % int(360 // bin_size)


def angle_to_colour(bin_idx, n_bins=12):
    """Convert bin index to HSV color with high saturation."""
    hue = bin_idx / n_bins
    return colorsys.hsv_to_rgb(hue, 0.9, 0.9)


def bin_and_average_states(states, metadata, n_bins):
    """Bin states by cued color and average within bins.
    
    Args:
        states: [N_trials, D] neural state array
        metadata: [N_trials, 3] with columns [cue, color1_angle, color2_angle]
        n_bins: number of angle bins
        
    Returns:
        averaged: dict {cue -> [n_bins, D]} with NaN for empty bins
    """
    bin_size = 360.0 / n_bins
    binned = {1: {b: [] for b in range(n_bins)}, 2: {b: [] for b in range(n_bins)}}
    
    for i, (cue, c1, c2) in enumerate(metadata):
        cue = int(cue)
        cued_angle = c1 if cue == 1 else c2
        b = bin_angle(cued_angle, bin_size)
        binned[cue][b].append(states[i])
    
    averaged = {
        1: np.full((n_bins, states.shape[1]), np.nan, dtype=np.float32),
        2: np.full((n_bins, states.shape[1]), np.nan, dtype=np.float32),
    }
    
    for cue in [1, 2]:
        for b in range(n_bins):
            if binned[cue][b]:
                averaged[cue][b] = np.mean(binned[cue][b], axis=0)
    
    return averaged


def make_full_step_trajectory_dataset(states_seq_by_prep, metadata, prep_indices, n_bins):
    """Build pooled points over all preparatory timesteps for global PCA.
    
    Args:
        states_seq_by_prep: dict {prep_idx -> list of [T_prep, D] sequences}
        metadata: [N_trials, 3]
        prep_indices: list of prep indices to include
        n_bins: number of angle bins
        
    Returns:
        points: [M, D] pooled neural points
        labels: [M, 6] with columns [prep_idx, cue, bin, prep_order, local_step, global_step]
        prep_step_counts: dict {prep_idx -> T_prep}
    """
    points = []
    labels = []
    prep_step_counts = {}
    global_step_offset = 0
    
    for prep_order, p in enumerate(prep_indices):
        seqs = states_seq_by_prep[p]
        if len(seqs) == 0:
            prep_step_counts[int(p)] = 0
            continue
        
        T_p = seqs[0].shape[0]
        prep_step_counts[int(p)] = int(T_p)
        
        for local_step in range(T_p):
            step_states = np.stack([trial_seq[local_step] for trial_seq in seqs], axis=0)
            averaged = bin_and_average_states(step_states, metadata, n_bins=n_bins)
            
            for cue in [1, 2]:
                for b in range(n_bins):
                    vec = averaged[cue][b]
                    if np.isnan(vec).any():
                        continue
                    points.append(vec)
                    labels.append([int(p), int(cue), int(b), int(prep_order), 
                                  int(local_step), int(global_step_offset + local_step)])
        
        global_step_offset += T_p
    
    return np.asarray(points), np.asarray(labels, dtype=int), prep_step_counts


def make_full_step_diffusion_dataset(states_seq_by_diffusion, metadata, n_bins):
    """Build pooled points over all diffusion timesteps for global PCA.
    
    Args:
        states_seq_by_diffusion: [N_trials, T_diff, D]
        metadata: [N_trials, 3]
        n_bins: number of angle bins
        
    Returns:
        points: [M, D] pooled neural points
        labels: [M, 4] with columns [cue, bin, t, t] (t appears twice for consistency)
        T_diff: number of diffusion timesteps
    """
    points = []
    labels = []
    
    if states_seq_by_diffusion.shape[0] == 0:
        return (
            np.empty((0, states_seq_by_diffusion.shape[-1])),
            np.empty((0, 4), dtype=int),
            0,
        )
    
    T_d = int(states_seq_by_diffusion.shape[1])
    for t in range(T_d):
        step_states = states_seq_by_diffusion[:, t, :]
        averaged = bin_and_average_states(step_states, metadata, n_bins=n_bins)
        
        for cue in [1, 2]:
            for b in range(n_bins):
                vec = averaged[cue][b]
                if np.isnan(vec).any():
                    continue
                points.append(vec)
                labels.append([int(cue), int(b), int(t), int(t)])
    
    return np.asarray(points), np.asarray(labels, dtype=int), T_d


def compute_3d_limits(pca_coords):
    """Compute padded axis limits for a 3D PCA projection."""
    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    return [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]


def plot_global_3d_trajectories(
    pca_coords,
    labels,
    pca,
    n_bins,
    out_path,
    title,
    label_schema='prep',
    lims=None,
    center_coords=False,
):
    """Plot 3D trajectories in PCA space with bin-colored paths.
    
    Args:
        pca_coords: [M, 3] PCA-projected points
        labels: [M, 6] or [M, 4] label array (depends on label_schema)
        pca: fitted PCA object
        n_bins: number of bins (for hue coloring)
        out_path: path to save figure
        title: plot title
        label_schema: 'prep' or 'diffusion' (determines label column mapping)
        lims: optional precomputed axis limits to reuse across figures
        center_coords: whether to recenter projected coordinates before plotting
    """
    coords = np.array(pca_coords, copy=True)
    if center_coords:
        coords = coords - coords.mean(axis=0, keepdims=True)

    if lims is None:
        lims = compute_3d_limits(coords)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Map labels columns based on schema
    if label_schema == 'prep':
        cue_col, bin_col, time_col = 1, 2, 5
    else:  # diffusion
        cue_col, bin_col, time_col = 0, 1, 3
    
    cues = np.unique(labels[:, cue_col])
    bins = np.unique(labels[:, bin_col])
    
    # Plot trajectories: one trajectory per (cue, bin) pair
    for cue in cues:
        marker = 'o' if cue == 1 else '^'
        for b in bins:
            m = (labels[:, cue_col] == cue) & (labels[:, bin_col] == b)
            if not np.any(m):
                continue
            
            idx = np.where(m)[0]
            idx = idx[np.argsort(labels[idx, time_col])]
            traj = coords[idx]
            color = angle_to_colour(int(b), n_bins=n_bins)
            
            # Draw line if trajectory has multiple points
            if traj.shape[0] >= 2:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       color=color, alpha=0.55, linewidth=1.8)
            
            # Plot points and make the endpoints stand out.
            ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], 
                      c=[color], marker=marker, s=36, alpha=0.9, 
                      edgecolors='k', linewidths=0.4)

            if traj.shape[0] >= 1:
                ax.scatter(
                    traj[0, 0], traj[0, 1], traj[0, 2],
                    facecolors='white', marker='o', s=180, alpha=1.0,
                    edgecolors='k', linewidths=2.2,
                )
            if traj.shape[0] >= 2:
                ax.scatter(
                    traj[-1, 0], traj[-1, 1], traj[-1, 2],
                    facecolors='black', marker='o', s=95, alpha=1.0,
                    edgecolors='white', linewidths=2.2,
                )
    
    # Set limits and labels
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}\\%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}\\%)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}\\%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.25)

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', color='w', markerfacecolor='lightgray', markeredgecolor='k', markersize=7, label='Cue 1'),
        Line2D([0], [0], marker='^', linestyle='None', color='w', markerfacecolor='lightgray', markeredgecolor='k', markersize=7, label='Cue 2'),
        Line2D([0], [0], marker='o', linestyle='None', color='w', markerfacecolor='white', markeredgecolor='k', markersize=9, label='Start'),
        Line2D([0], [0], marker='o', linestyle='None', color='w', markerfacecolor='black', markeredgecolor='white', markersize=6, label='End'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    
    legend_labels = ['Cue 1', 'Cue 2', 'Start', 'End']
    plt.tight_layout()
    save_figure(fig, out_path)
    save_legend(legend_handles, legend_labels, Path(out_path))
    plt.close(fig)
    print('Saved:', out_path)
