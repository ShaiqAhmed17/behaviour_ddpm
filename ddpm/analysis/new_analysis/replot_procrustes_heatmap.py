#!/usr/bin/env python3
"""Replot Procrustes heatmap from saved .npy data with publication style and outliers removed."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.vis.style import set_publication_style, save_figure

set_publication_style()
import matplotlib.pyplot as plt

DATA_DIR   = REPO_ROOT / 'ddpm/analysis/new_analysis/results/procrustes_heatmap_15dirs_all_students'
OUTPUT_DIR = DATA_DIR

# ── Exclusions (consistent with MDS.py and MDS_wasserstein.py) ───────────────
EXCLUDE_TEACHER_ROWS = {
    'dir_00', 'dir_02', 'dir_03', 'dir_06', 'dir_08',
    'dir_09', 'dir_10', 'dir_11',
}
EXCLUDE_STUDENT_COLS = {
    'abl0_0', 'abl0_1',   # dir 0
    'abl2_0',              # dir 2
    'abl3_0',              # dir 3
    'abl6_0',              # dir 6
    'abl9_0',              # dir 9
    'abl10_0',             # dir 10
    'abl11_0', 'abl11_1', # dir 11
    'multi_0-1_s0',        # Wasserstein MDS outlier
}


def _filter(D, row_labels, col_labels):
    row_mask = [l not in EXCLUDE_TEACHER_ROWS for l in row_labels]
    col_mask = [l not in EXCLUDE_STUDENT_COLS for l in col_labels]
    row_idx  = [i for i, keep in enumerate(row_mask) if keep]
    col_idx  = [i for i, keep in enumerate(col_mask) if keep]
    D_f      = D[np.ix_(row_idx, col_idx)]
    r_labs   = [row_labels[i] for i in row_idx]
    c_labs   = [col_labels[i] for i in col_idx]
    return D_f, r_labs, c_labs


def _plot_heatmap(D, row_labels, col_labels, title, out_stem, vmin=None, vmax=None):
    n_t, n_s = D.shape
    cell = 0.75
    fig_w = max(8, cell * n_s + 2.5)
    fig_h = max(4, cell * n_t + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(D, aspect='auto', cmap='viridis_r', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Procrustes residual (a.u.)', fontsize=13)
    cb.ax.tick_params(labelsize=11)

    ax.set_xticks(range(n_s))
    ax.set_xticklabels(col_labels, rotation=40, ha='right', fontsize=10)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel('Student', fontsize=14)
    ax.set_ylabel('Teacher', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    mid = (D.max() + D.min()) / 2
    for i in range(n_t):
        for j in range(n_s):
            ax.text(j, i, f'{D[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if D[i, j] < mid else 'black')

    fig.tight_layout()
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f'Saved: {out_stem}.pdf / .png')


def _plot_comparison(D_scaled, D_fixed, row_labels, col_labels, out_stem):
    vmin = min(D_scaled.min(), D_fixed.min())
    vmax = max(D_scaled.max(), D_fixed.max())
    n_t, n_s = D_scaled.shape
    cell = 0.75

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * (cell * n_s + 2.5), cell * n_t + 2.5),
        sharey=True,
    )
    for ax, D, subtitle in zip(axes,
                                [D_scaled, D_fixed],
                                ['Free scale ($c$ optimised)', 'Fixed scale ($c = 1$)']):
        im = ax.imshow(D, aspect='auto', cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_s))
        ax.set_xticklabels(col_labels, rotation=40, ha='right', fontsize=9)
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_xlabel('Student', fontsize=13)
        ax.set_title(subtitle, fontsize=14, fontweight='bold')
        mid = (vmax + vmin) / 2
        for i in range(n_t):
            for j in range(n_s):
                ax.text(j, i, f'{D[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if D[i, j] < mid else 'black')

    axes[0].set_ylabel('Teacher', fontsize=13)
    fig.suptitle('Procrustes residual: scaling comparison', fontsize=16, fontweight='bold')
    cb = fig.colorbar(im, ax=axes, label='Procrustes residual (a.u.)', shrink=0.8)
    cb.ax.tick_params(labelsize=10)
    fig.tight_layout()
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f'Saved: {out_stem}.pdf / .png')


# ── Load ─────────────────────────────────────────────────────────────────────
lab        = np.load(DATA_DIR / 'labels.npz', allow_pickle=True)
t_labels   = [str(x) for x in lab['teacher_labels']]
s_labels   = [str(x) for x in lab['student_labels']]
D_scaled   = np.load(DATA_DIR / 'heatmap_scaled.npy')
D_fixed    = np.load(DATA_DIR / 'heatmap_fixed.npy')

print(f'Loaded: {D_scaled.shape}  ({len(t_labels)} teachers × {len(s_labels)} students)')

# ── Filter ───────────────────────────────────────────────────────────────────
D_sc_f, t_f, s_f = _filter(D_scaled, t_labels, s_labels)
D_fx_f, _,   _   = _filter(D_fixed,  t_labels, s_labels)
print(f'After filtering: {D_sc_f.shape}  ({len(t_f)} teachers × {len(s_f)} students)')

vmin = min(D_sc_f.min(), D_fx_f.min())
vmax = max(D_sc_f.max(), D_fx_f.max())

# ── Plot ─────────────────────────────────────────────────────────────────────
_plot_heatmap(D_sc_f, t_f, s_f,
              title='Procrustes residual (free scale)',
              out_stem=OUTPUT_DIR / 'heatmap_scaled', vmin=vmin, vmax=vmax)

_plot_heatmap(D_fx_f, t_f, s_f,
              title='Procrustes residual (fixed scale)',
              out_stem=OUTPUT_DIR / 'heatmap_fixed', vmin=vmin, vmax=vmax)

_plot_comparison(D_sc_f, D_fx_f, t_f, s_f,
                 out_stem=OUTPUT_DIR / 'heatmap_comparison')

print('Done.')
