"""
Appends the no_ablation student as a 20th column to the existing 15×19 heatmaps.
All trajectories are already cached — no model extraction needed.
"""

import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.analysis.new_analysis.run_procrustes_heatmap import (
    _load_nullspace, _generate_trials, save_heatmap_diagnostics,
    plot_heatmap, plot_comparison_heatmaps,
)
from ddpm.analysis.new_analysis.procrustes_heatmap import (
    project_to_nullspace, compute_heatmap,
)

TEACHER_RUN  = 'index_cued_first_diffusion_0.3_swap_7'
RESULTS_ROOT = REPO_ROOT / 'results_link_sampler'
CACHE_DIR    = REPO_ROOT / 'ddpm/analysis/new_analysis/results/procrustes_heatmap/traj_cache'
OUT          = REPO_ROOT / 'ddpm/analysis/new_analysis/results/procrustes_heatmap_15dirs_all_students'
DEVICE       = 'cuda'
S            = 512

# ---------------------------------------------------------------------------
# Load existing matrices and labels
# ---------------------------------------------------------------------------
D_scaled = np.load(OUT / 'heatmap_scaled.npy')   # (15, 19)
D_fixed  = np.load(OUT / 'heatmap_fixed.npy')    # (15, 19)

saved = np.load(OUT / 'labels.npz')
teacher_labels = saved['teacher_labels'].tolist()
student_labels = saved['student_labels'].tolist()

print(f'Existing matrix: {D_scaled.shape}  teachers={teacher_labels}')

# ---------------------------------------------------------------------------
# Load nullspace and trials (must match original run)
# ---------------------------------------------------------------------------
nullspace = _load_nullspace(RESULTS_ROOT / TEACHER_RUN)
trials = _generate_trials(angle_step=30)
trial_indices = list(range(len(trials)))

# ---------------------------------------------------------------------------
# Load 15 teacher trajectories from full cache
# ---------------------------------------------------------------------------
def load_teacher(direction_idx):
    abl = 'unablated' if direction_idx is None else f'abl{direction_idx:02d}'
    return np.load(CACHE_DIR / f'{TEACHER_RUN}_{abl}_full_S{S}.npy').astype(np.float32)

teacher_trajs_16d = [load_teacher(None)] + [load_teacher(i) for i in range(14)]
teacher_trajs = [project_to_nullspace(t, nullspace).astype(np.float32) for t in teacher_trajs_16d]
print(f'Loaded {len(teacher_trajs)} teacher trajectories. Shape: {teacher_trajs[0].shape}')

# ---------------------------------------------------------------------------
# Load no_ablation student from full cache
# ---------------------------------------------------------------------------
no_abl_run = 'index_cued_first_diffusion_0.3_swap_recovery_ablation_no_ablation_0'
no_abl_traj = np.load(CACHE_DIR / f'{no_abl_run}_unablated_full_S{S}.npy').astype(np.float32)
no_abl_traj = project_to_nullspace(no_abl_traj, nullspace).astype(np.float32)
print(f'Loaded no_ablation student. Shape: {no_abl_traj.shape}')

# ---------------------------------------------------------------------------
# Compute the one missing column — scaled and fixed
# ---------------------------------------------------------------------------
print('\nComputing 15 × 1 column (allow_scaling=True) ...')
col_scaled = compute_heatmap(
    teacher_trajs, [no_abl_traj],
    allow_scaling=True, n_restarts=3, max_iter=50, tol=1e-5,
    n_jobs_lap=-1, seed=42, device=DEVICE, match_weights=None,
)

print('\nComputing 15 × 1 column (allow_scaling=False) ...')
col_fixed = compute_heatmap(
    teacher_trajs, [no_abl_traj],
    allow_scaling=False, n_restarts=3, max_iter=50, tol=1e-5,
    n_jobs_lap=-1, seed=42, device=DEVICE, match_weights=None,
)

# ---------------------------------------------------------------------------
# Append column and save
# ---------------------------------------------------------------------------
new_col_s = col_scaled['residuals']   # (15, 1)
new_col_f = col_fixed['residuals']    # (15, 1)

D_scaled_new = np.concatenate([D_scaled, new_col_s], axis=1)  # (15, 20)
D_fixed_new  = np.concatenate([D_fixed,  new_col_f], axis=1)

student_labels_new = student_labels + ['no_ablation']

np.save(OUT / 'heatmap_scaled.npy', D_scaled_new)
np.save(OUT / 'heatmap_fixed.npy',  D_fixed_new)
np.savez(OUT / 'labels.npz',
         teacher_labels=np.array(teacher_labels),
         student_labels=np.array(student_labels_new))
print(f'\nUpdated matrices: {D_scaled_new.shape}')

# Merge diagnostics
diag_s = np.load(OUT / 'diagnostics_scaled.npz')
diag_f = np.load(OUT / 'diagnostics_fixed.npz')

def append_diag(diag, col_result):
    r = col_result['alignment_results']
    n_t = len(r)
    return dict(
        residuals          = np.concatenate([diag['residuals'],          col_result['residuals']], axis=1),
        identity_residuals = np.concatenate([diag['identity_residuals'], np.array([[r[i][0].identity_residual] for i in range(n_t)])], axis=1),
        c_values           = np.concatenate([diag['c_values'],           np.array([[r[i][0].c] for i in range(n_t)])], axis=1),
        restart_residuals  = np.concatenate([diag['restart_residuals'],  np.array([[[v for v in r[i][0].restart_residuals]] for i in range(n_t)])], axis=1),
        trace_lengths      = np.concatenate([diag['trace_lengths'],      np.array([[len(r[i][0].objective_trace)] for i in range(n_t)])], axis=1),
    )

np.savez(OUT / 'diagnostics_scaled.npz', **append_diag(diag_s, col_scaled))
np.savez(OUT / 'diagnostics_fixed.npz',  **append_diag(diag_f, col_fixed))

# ---------------------------------------------------------------------------
# Replot
# ---------------------------------------------------------------------------
vmin = min(D_scaled_new.min(), D_fixed_new.min())
vmax = max(D_scaled_new.max(), D_fixed_new.max())

plot_heatmap(D_scaled_new, teacher_labels, student_labels_new,
             title='Procrustes residual — 15 teacher dirs × 20 students (free scale)',
             out_path=OUT / 'heatmap_scaled.png', vmin=vmin, vmax=vmax)
plot_heatmap(D_fixed_new, teacher_labels, student_labels_new,
             title='Procrustes residual — 15 teacher dirs × 20 students (fixed scale)',
             out_path=OUT / 'heatmap_fixed.png', vmin=vmin, vmax=vmax)
plot_comparison_heatmaps(D_scaled_new, D_fixed_new, teacher_labels, student_labels_new,
                         out_path=OUT / 'heatmap_comparison.png')

print('\nDone. no_ablation column appended and plots regenerated.')
print(f'Scaled col: {new_col_s.flatten().round(4)}')
print(f'Fixed  col: {new_col_f.flatten().round(4)}')
