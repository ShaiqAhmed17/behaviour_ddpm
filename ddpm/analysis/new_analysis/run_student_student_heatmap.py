"""
run_student_student_heatmap.py

20×20 student × student Procrustes residual heatmap.
All 20 student full trajectories are already cached — no model extraction.
Projected through the teacher's nullspace for a common reference frame.

Rows/columns are sorted by ablation direction index so same-direction
students are adjacent, making block structure visible.

Output: ddpm/analysis/new_analysis/results/student_student_heatmap/
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.analysis.new_analysis.run_procrustes_heatmap import _load_nullspace
from ddpm.analysis.new_analysis.procrustes_heatmap import (
    project_to_nullspace, compute_heatmap,
)

TEACHER_RUN  = 'index_cued_first_diffusion_0.3_swap_7'
RESULTS_ROOT = REPO_ROOT / 'results_link_sampler'
CACHE_DIR    = REPO_ROOT / 'ddpm/analysis/new_analysis/results/procrustes_heatmap/traj_cache'
OUT          = REPO_ROOT / 'ddpm/analysis/new_analysis/results/student_student_heatmap'
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda'
S      = 512

# ---------------------------------------------------------------------------
# Student list — sorted by (direction, seed) so same-direction students
# are adjacent in the heatmap.
# Each entry: (run_name, short_label, direction_or_None)
# ---------------------------------------------------------------------------
STUDENTS = [
    # dir 0
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_0_0',  'abl0_s0',  0),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_0_1',  'abl0_s1',  0),
    # dir 1
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_1_0',  'abl1_s0',  1),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_1_1',  'abl1_s1',  1),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_1_2',  'abl1_s2',  1),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_1_3',  'abl1_s3',  1),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_1_4',  'abl1_s4',  1),
    # dirs 2–7, 9–11, 13 (one seed each)
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_2_0',  'abl2_s0',  2),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_3_0',  'abl3_s0',  3),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_4_0',  'abl4_s0',  4),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_5_0',  'abl5_s0',  5),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_6_0',  'abl6_s0',  6),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_7_0',  'abl7_s0',  7),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_9_0',  'abl9_s0',  9),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_10_0', 'abl10_s0', 10),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_11_0', 'abl11_s0', 11),
    # dir 12
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_12_0', 'abl12_s0', 12),
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_12_1', 'abl12_s1', 12),
    # dir 13
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_13_0', 'abl13_s0', 13),
    # no ablation
    ('index_cued_first_diffusion_0.3_swap_recovery_ablation_no_ablation_0', 'no_abl', None),
]

labels    = [s[1] for s in STUDENTS]
dir_ids   = [s[2] for s in STUDENTS]
run_names = [s[0] for s in STUDENTS]
N = len(STUDENTS)

# ---------------------------------------------------------------------------
# Load nullspace and trajectories (all from cache)
# ---------------------------------------------------------------------------
nullspace = _load_nullspace(RESULTS_ROOT / TEACHER_RUN)
print(f'Nullspace: {nullspace.shape}')

trajs = []
for run_name, label, _ in STUDENTS:
    path = CACHE_DIR / f'{run_name}_unablated_full_S{S}.npy'
    t = np.load(path).astype(np.float32)
    t = project_to_nullspace(t, nullspace).astype(np.float32)
    trajs.append(t)
    print(f'  loaded {label}  {t.shape}')

# ---------------------------------------------------------------------------
# Compute N×N heatmap — scaled
# ---------------------------------------------------------------------------
print(f'\nComputing {N}×{N} student-student heatmap (allow_scaling=True) ...')
result_scaled = compute_heatmap(
    trajs, trajs,
    allow_scaling=True, n_restarts=3, max_iter=50, tol=1e-5,
    n_jobs_lap=-1, seed=42, device=DEVICE, match_weights=None,
)
D_scaled = result_scaled['residuals']
np.save(OUT / 'heatmap_scaled.npy', D_scaled)
print(f'Saved heatmap_scaled.npy  min={D_scaled.min():.4f}  max={D_scaled.max():.4f}')

# ---------------------------------------------------------------------------
# Compute N×N heatmap — fixed scale
# ---------------------------------------------------------------------------
print(f'\nComputing {N}×{N} student-student heatmap (allow_scaling=False) ...')
result_fixed = compute_heatmap(
    trajs, trajs,
    allow_scaling=False, n_restarts=3, max_iter=50, tol=1e-5,
    n_jobs_lap=-1, seed=42, device=DEVICE, match_weights=None,
)
D_fixed = result_fixed['residuals']
np.save(OUT / 'heatmap_fixed.npy', D_fixed)
np.savez(OUT / 'labels.npz', labels=np.array(labels), dir_ids=np.array(dir_ids, dtype=object))
print(f'Saved heatmap_fixed.npy  min={D_fixed.min():.4f}  max={D_fixed.max():.4f}')


# ---------------------------------------------------------------------------
# Plotting helper — draws boxes around same-direction blocks
# ---------------------------------------------------------------------------
def plot_student_heatmap(D, labels, dir_ids, title, out_path, vmin=None, vmax=None):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * n + 2), max(9, 0.55 * n + 2)))

    # Mask diagonal (self-alignment is trivially 0)
    D_plot = D.copy().astype(float)
    np.fill_diagonal(D_plot, np.nan)

    if vmin is None: vmin = np.nanpercentile(D_plot, 2)
    if vmax is None: vmax = np.nanpercentile(D_plot, 98)

    im = ax.imshow(D_plot, aspect='equal', cmap='viridis_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Procrustes residual', shrink=0.7)

    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, pad=10)

    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f'{D[i,j]:.2f}', ha='center', va='center',
                        fontsize=5, color='white' if D[i,j] < (vmin+vmax)/2 else 'black')

    # Draw rectangles around same-direction blocks
    visited = set()
    for di in set(d for d in dir_ids if d is not None):
        idxs = [k for k, d in enumerate(dir_ids) if d == di]
        if len(idxs) < 2:
            continue
        if di in visited:
            continue
        visited.add(di)
        lo, hi = min(idxs), max(idxs)
        rect = mpatches.Rectangle(
            (lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
            linewidth=2, edgecolor='crimson', facecolor='none', zorder=5,
        )
        ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# ---------------------------------------------------------------------------
# Plot both
# ---------------------------------------------------------------------------
off_diag_mask = ~np.eye(N, dtype=bool)
vmin = min(np.nanpercentile(D_scaled[off_diag_mask], 2),
           np.nanpercentile(D_fixed[off_diag_mask],  2))
vmax = max(np.nanpercentile(D_scaled[off_diag_mask], 98),
           np.nanpercentile(D_fixed[off_diag_mask],  98))

plot_student_heatmap(D_scaled, labels, dir_ids,
    title='Student × student Procrustes residual (free scale)\nRed boxes = same ablation direction',
    out_path=OUT / 'heatmap_scaled.png', vmin=vmin, vmax=vmax)

plot_student_heatmap(D_fixed, labels, dir_ids,
    title='Student × student Procrustes residual (fixed scale)\nRed boxes = same ablation direction',
    out_path=OUT / 'heatmap_fixed.png', vmin=vmin, vmax=vmax)

# ---------------------------------------------------------------------------
# Summary statistics: within-direction vs between-direction off-diagonal
# ---------------------------------------------------------------------------
within, between = [], []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if dir_ids[i] is not None and dir_ids[i] == dir_ids[j]:
            within.append(D_scaled[i, j])
        else:
            between.append(D_scaled[i, j])

print(f'\nScaled residual summary (off-diagonal):')
print(f'  Within-direction  (n={len(within)}):  mean={np.mean(within):.4f}  median={np.median(within):.4f}')
print(f'  Between-direction (n={len(between)}): mean={np.mean(between):.4f}  median={np.median(between):.4f}')
print(f'\nOutput: {OUT}')
