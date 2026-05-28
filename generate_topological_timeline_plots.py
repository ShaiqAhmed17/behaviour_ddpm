#!/usr/bin/env python3
"""
Generate timeline plots for topological features (same format as ring geometry plots).
For each teacher/student: create cue1_vs_cue2 and target_vs_distractor plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from MDS import extract_all_features

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_DIR = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'

N_BINS = 12
BIN_SIZE = 360.0 / N_BINS

def _bangle(a, sz=30.0):
    return int(a // sz) % int(360 // sz)

def _group_bins(tl_mt):
    """Group trial indices by cue and target/distractor angle."""
    g = {c: {'target': {b: [] for b in range(N_BINS)},
              'distractor': {b: [] for b in range(N_BINS)}} for c in [1, 2]}
    for i, (cue, c1, c2) in enumerate(tl_mt):
        cue = int(cue)
        ta, da = (c1, c2) if cue == 1 else (c2, c1)
        g[cue]['target'][int(ta // BIN_SIZE) % N_BINS].append(i)
        g[cue]['distractor'][int(da // BIN_SIZE) % N_BINS].append(i)
    for c in [1, 2]:
        for role in ['target', 'distractor']:
            for b in range(N_BINS):
                g[c][role][b] = np.array(g[c][role][b], dtype=np.int64)
    return g

def _bin_avg(tl_st, idx_per_bin):
    """Average neural states within each angle bin."""
    T, D = tl_st.shape[1], tl_st.shape[2]
    out = np.zeros((N_BINS, T, D), np.float32)
    for b, idx in idx_per_bin.items():
        if len(idx):
            out[b] = tl_st[idx].mean(0)
    return out

def compute_topological_features_timewise(tl_st_binned):
    """
    Compute topological features for binned neural states (N_BINS, T, D).
    Returns dict of features where each is (T,) array.
    """
    features_over_time = {}
    T = tl_st_binned.shape[1]

    for t in range(T):
        X = tl_st_binned[:, t, :].astype(np.float32)  # (N_BINS, D)

        # PCA for dimensionality
        pca = PCA()
        pca.fit(X)

        # 1. Intrinsic dimensionality (% variance at 90%)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        id_90 = float(np.argmax(cumsum_var >= 0.9) + 1) / len(pca.explained_variance_ratio_)
        features_over_time.setdefault('intrinsic_dim_90pct', []).append(id_90)

        # 2. Participation ratio
        lam = pca.explained_variance_
        part_ratio = float((lam.sum() ** 2) / (lam ** 2).sum())
        features_over_time.setdefault('participation_ratio', []).append(part_ratio)

        # 3. Variance explained by top-3 PCs
        var_top3 = float(pca.explained_variance_ratio_[:3].sum())
        features_over_time.setdefault('var_top3_pcs', []).append(var_top3)

        # 4. Reconstruction error at rank-3
        X_centered = X - X.mean(0)
        pca3 = PCA(n_components=3)
        X_recon = pca3.inverse_transform(pca3.fit_transform(X_centered))
        per_sample_err = np.sum((X_centered - X_recon) ** 2, axis=1)
        recon_err = float(np.sqrt(per_sample_err.mean()))
        features_over_time.setdefault('reconstruction_error_rank3', []).append(recon_err)

    # Convert lists to arrays
    return {k: np.array(v, dtype=np.float32) for k, v in features_over_time.items()}

def plot_topological_timeline(npz_path, output_dir, plot_prefix=''):
    """Generate topological timeline plots for a model."""

    try:
        data = np.load(npz_path)
        tl_st = data['tl_st']  # (n_trials, T, D)
        tl_mt = data['tl_mt']  # (n_trials, 3): cue, c1, c2
        data.close()
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return False

    # Group by cue and target/distractor
    groups = _group_bins(tl_mt)

    # Compute topological features for cue1 target vs distractor
    tgt_bin_c1 = _bin_avg(tl_st, groups[1]['target'])
    dis_bin_c1 = _bin_avg(tl_st, groups[1]['distractor'])

    topo_tgt_c1 = compute_topological_features_timewise(tgt_bin_c1)
    topo_dis_c1 = compute_topological_features_timewise(dis_bin_c1)

    # Compute topological features for cue2 target vs distractor
    tgt_bin_c2 = _bin_avg(tl_st, groups[2]['target'])
    dis_bin_c2 = _bin_avg(tl_st, groups[2]['distractor'])

    topo_tgt_c2 = compute_topological_features_timewise(tgt_bin_c2)
    topo_dis_c2 = compute_topological_features_timewise(dis_bin_c2)

    # Compute for cue1 vs cue2
    cue1_bin = _bin_avg(tl_st, {b: np.concatenate([groups[1]['target'][b], groups[1]['distractor'][b]])
                                 for b in range(N_BINS)})
    cue2_bin = _bin_avg(tl_st, {b: np.concatenate([groups[2]['target'][b], groups[2]['distractor'][b]])
                                 for b in range(N_BINS)})

    topo_c1 = compute_topological_features_timewise(cue1_bin)
    topo_c2 = compute_topological_features_timewise(cue2_bin)

    T = tl_st.shape[1]
    time_ax = np.arange(T)

    # Plot 1: Cue1 target vs distractor
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(['intrinsic_dim_90pct', 'participation_ratio', 'var_top3_pcs', 'reconstruction_error_rank3']):
        ax = axes[i]
        ax.plot(time_ax, topo_tgt_c1[metric], 'o-', label='Target', color='steelblue', linewidth=2, markersize=4)
        ax.plot(time_ax, topo_dis_c1[metric], 's-', label='Distractor', color='coral', linewidth=2, markersize=4)
        ax.set_xlabel('Diffusion Step', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_title(f'{metric}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Topological Features: Cue1 Target vs Distractor {plot_prefix}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'timeline_topological_cue1_target_vs_distractor.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Plot 2: Cue2 target vs distractor
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(['intrinsic_dim_90pct', 'participation_ratio', 'var_top3_pcs', 'reconstruction_error_rank3']):
        ax = axes[i]
        ax.plot(time_ax, topo_tgt_c2[metric], 'o-', label='Target', color='steelblue', linewidth=2, markersize=4)
        ax.plot(time_ax, topo_dis_c2[metric], 's-', label='Distractor', color='coral', linewidth=2, markersize=4)
        ax.set_xlabel('Diffusion Step', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_title(f'{metric}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Topological Features: Cue2 Target vs Distractor {plot_prefix}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'timeline_topological_cue2_target_vs_distractor.png', dpi=100, bbox_inches='tight')
    plt.close()

    # Plot 3: Cue1 vs Cue2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(['intrinsic_dim_90pct', 'participation_ratio', 'var_top3_pcs', 'reconstruction_error_rank3']):
        ax = axes[i]
        ax.plot(time_ax, topo_c1[metric], 'o-', label='Cue 1', color='steelblue', linewidth=2, markersize=4)
        ax.plot(time_ax, topo_c2[metric], 's-', label='Cue 2', color='coral', linewidth=2, markersize=4)
        ax.set_xlabel('Diffusion Step', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_title(f'{metric}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'Topological Features: Cue1 vs Cue2 {plot_prefix}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / f'timeline_topological_cue1_vs_cue2.png', dpi=100, bbox_inches='tight')
    plt.close()

    return True

# ─────────────────────────────────────────────────────────────────────────────
# Main: Process all teachers and students
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("GENERATING TOPOLOGICAL TIMELINE PLOTS")
print("=" * 100)

all_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])

# Teachers
teacher_dirs = [d for d in all_dirs if 'ablated_teacher_dir_' in d.name or d.name == 'index_cued_first_diffusion_0.3_swap_7']
print(f"\nProcessing {len(teacher_dirs)} teachers...")

for i, teacher_dir in enumerate(teacher_dirs):
    npz_path = teacher_dir / 'timeline_raw_states.npz'
    if npz_path.exists():
        print(f"  [{i+1}/{len(teacher_dirs)}] {teacher_dir.name}...", end=' ', flush=True)
        if plot_topological_timeline(npz_path, teacher_dir, ''):
            print("✓")
        else:
            print("✗")

# Students
student_dirs = [d for d in all_dirs if 'index_cued_first_diffusion_0.3_swap_recovery_ablation_' in d.name]
print(f"\nProcessing {len(student_dirs)} students...")

for i, student_dir in enumerate(student_dirs):
    npz_path = student_dir / 'timeline_raw_states.npz'
    if npz_path.exists():
        print(f"  [{i+1}/{len(student_dirs)}] {student_dir.name}...", end=' ', flush=True)
        if plot_topological_timeline(npz_path, student_dir, ''):
            print("✓")
        else:
            print("✗")

print("\n" + "=" * 100)
print("✓ COMPLETE")
print("=" * 100)
