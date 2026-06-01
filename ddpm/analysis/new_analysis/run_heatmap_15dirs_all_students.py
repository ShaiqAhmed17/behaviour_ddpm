"""
run_heatmap_15dirs_all_students.py

15 teacher rows (unablated + abl00..abl13) × all student runs, including:
  - single nullspace-direction:  ...recovery_ablation_{N}_{seed}
  - multi  nullspace-direction:  ...recovery_ablation_{N}-{M}-..._{seed}
  - PCA-based:                   ...recovery_pca_ablation_{N}(-{M}...?)_{seed}
  - no-ablation:                 ...recovery_no_ablation_{seed}

Output: ddpm/analysis/new_analysis/results/procrustes_heatmap_15dirs_all_students/
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.analysis.new_analysis.run_procrustes_heatmap import (
    load_or_extract_trajectories,
    save_heatmap_diagnostics,
    plot_heatmap,
    plot_comparison_heatmaps,
    plot_permutation_test,
    _load_nullspace,
    _generate_trials,
)
from ddpm.analysis.new_analysis.procrustes_heatmap import (
    project_to_nullspace,
    compute_heatmap,
    permutation_test,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEACHER_RUN   = 'index_cued_first_diffusion_0.3_swap_7'
RESULTS_ROOT  = REPO_ROOT / 'results_link_sampler'
CACHE_DIR     = REPO_ROOT / 'ddpm/analysis/new_analysis/results/procrustes_heatmap/traj_cache'
OUTPUT_DIR    = REPO_ROOT / 'ddpm/analysis/new_analysis/results/procrustes_heatmap_15dirs_all_students'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE        = 'cuda'
N_SAMPLES     = 512
N_RESTARTS    = 3
MAX_ITER      = 50
TOL           = 1e-5
N_PERMS       = 10_000
SEED          = 42
CHUNK_SIZE    = 64
INCLUDE_DIFFUSION = True    # full prep + diffusion trajectories (_full cache suffix)
MATCH_WEIGHTS = None        # uniform weighting across all timesteps

# ---------------------------------------------------------------------------
# Discover all student runs (single-dir, multi-dir, PCA, no-ablation)
# ---------------------------------------------------------------------------
_PAT_SINGLE = re.compile(
    r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(\d+)_(\d+)$'
)
_PAT_MULTI = re.compile(
    r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(\d+(?:-\d+)+)_(\d+)$'
)
_PAT_PCA = re.compile(
    r'^index_cued_first_diffusion_0\.3_swap_recovery_pca_ablation_(\d+(?:-\d+)*)_(\d+)$'
)
_PAT_NOABL = re.compile(
    r'^index_cued_first_diffusion_0\.3_swap_recovery_no_ablation_(\d+)$'
)

def _student_sort_key(name):
    for pat in [_PAT_SINGLE, _PAT_MULTI, _PAT_PCA, _PAT_NOABL]:
        m = pat.match(name)
        if m:
            groups = m.groups()
            # Sort by first numeric component, then remaining
            first = int(groups[0].split('-')[0]) if groups[0].replace('-','').isdigit() else 999
            seed  = int(groups[-1]) if groups[-1].isdigit() else 0
            return (first, seed, name)
    return (9999, 0, name)

def _is_student(d):
    if not d.is_dir():
        return False
    if not (d / 'state.mdl').exists() or not (d / 'args.yaml').exists():
        return False
    name = d.name
    return any(p.match(name) for p in [_PAT_SINGLE, _PAT_MULTI, _PAT_PCA, _PAT_NOABL])

student_runs = sorted(
    [d.name for d in RESULTS_ROOT.iterdir() if _is_student(d)],
    key=_student_sort_key,
)

print(f'Found {len(student_runs)} student runs:')
for r in student_runs:
    print(f'  {r}')

# ---------------------------------------------------------------------------
# Generate trials (angle_step=30 → 288 trials, matching existing caches)
# ---------------------------------------------------------------------------
trials = _generate_trials(angle_step=30)
trial_indices = list(range(len(trials)))
print(f'\nTrials: {len(trial_indices)}')

# ---------------------------------------------------------------------------
# Load teacher nullspace
# ---------------------------------------------------------------------------
nullspace = _load_nullspace(RESULTS_ROOT / TEACHER_RUN)
print(f'Nullspace: {nullspace.shape}')

# ---------------------------------------------------------------------------
# Teacher trajectories: unablated (row 0) + abl00..abl13 (rows 1..14)
# ---------------------------------------------------------------------------
teacher_trajs_16d = []
teacher_labels    = []

print('\nLoading unablated teacher ...')
t = load_or_extract_trajectories(
    RESULTS_ROOT / TEACHER_RUN, trials, DEVICE, N_SAMPLES,
    ablation_direction_idx=None,
    trial_indices=trial_indices,
    cache_dir=CACHE_DIR,
    label=TEACHER_RUN,
    chunk_size=CHUNK_SIZE,
    include_diffusion=INCLUDE_DIFFUSION,
)
teacher_trajs_16d.append(t)
teacher_labels.append('unablated')

for di in range(14):
    print(f'Loading teacher abl{di:02d} ...')
    t = load_or_extract_trajectories(
        RESULTS_ROOT / TEACHER_RUN, trials, DEVICE, N_SAMPLES,
        ablation_direction_idx=di,
        trial_indices=trial_indices,
        cache_dir=CACHE_DIR,
        label=TEACHER_RUN,
        chunk_size=CHUNK_SIZE,
        include_diffusion=INCLUDE_DIFFUSION,
    )
    teacher_trajs_16d.append(t)
    teacher_labels.append(f'dir_{di:02d}')

# ---------------------------------------------------------------------------
# Student trajectories
# ---------------------------------------------------------------------------
student_trajs_16d = []
student_labels    = []

for s_run in student_runs:
    print(f'Loading student: {s_run} ...')
    t = load_or_extract_trajectories(
        RESULTS_ROOT / s_run, trials, DEVICE, N_SAMPLES,
        ablation_direction_idx=None,
        trial_indices=trial_indices,
        cache_dir=CACHE_DIR,
        label=s_run,
        chunk_size=CHUNK_SIZE,
        include_diffusion=INCLUDE_DIFFUSION,
    )
    student_trajs_16d.append(t)
    # Generate a short label from whichever pattern matches
    m_s = _PAT_SINGLE.match(s_run)
    m_m = _PAT_MULTI.match(s_run)
    m_p = _PAT_PCA.match(s_run)
    m_n = _PAT_NOABL.match(s_run)
    if m_s:
        lbl = f'abl{m_s.group(1)}_{m_s.group(2)}'
    elif m_m:
        lbl = f'multi_{m_m.group(1)}_s{m_m.group(2)}'
    elif m_p:
        lbl = f'pca_{m_p.group(1)}_s{m_p.group(2)}'
    elif m_n:
        lbl = f'noabl_s{m_n.group(1)}'
    else:
        lbl = s_run
    student_labels.append(lbl)

# ---------------------------------------------------------------------------
# Project to 14-D nullspace
# ---------------------------------------------------------------------------
teacher_trajs = [project_to_nullspace(t, nullspace).astype(np.float32) for t in teacher_trajs_16d]
student_trajs = [project_to_nullspace(t, nullspace).astype(np.float32) for t in student_trajs_16d]
print(f'\nProjected to nullspace. Teacher shape: {teacher_trajs[0].shape}')

# ---------------------------------------------------------------------------
# Compute heatmap — scaled (c free)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print(f'Computing heatmap  {len(teacher_trajs)} teachers × {len(student_trajs)} students  (allow_scaling=True) ...')
print('='*60)
heatmap_scaled = compute_heatmap(
    teacher_trajs, student_trajs,
    allow_scaling=True,
    n_restarts=N_RESTARTS,
    max_iter=MAX_ITER,
    tol=TOL,
    n_jobs_lap=-1,
    seed=SEED,
    device=DEVICE,
    match_weights=MATCH_WEIGHTS,
)
D_scaled = heatmap_scaled['residuals']
np.save(OUTPUT_DIR / 'heatmap_scaled.npy', D_scaled)
save_heatmap_diagnostics(heatmap_scaled, OUTPUT_DIR / 'diagnostics_scaled.npz')

# ---------------------------------------------------------------------------
# Compute heatmap — fixed scale (c = 1)
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print('Computing heatmap  (allow_scaling=False) ...')
print('='*60)
heatmap_fixed = compute_heatmap(
    teacher_trajs, student_trajs,
    allow_scaling=False,
    n_restarts=N_RESTARTS,
    max_iter=MAX_ITER,
    tol=TOL,
    n_jobs_lap=-1,
    seed=SEED,
    device=DEVICE,
    match_weights=MATCH_WEIGHTS,
)
D_fixed = heatmap_fixed['residuals']
np.save(OUTPUT_DIR / 'heatmap_fixed.npy', D_fixed)
save_heatmap_diagnostics(heatmap_fixed, OUTPUT_DIR / 'diagnostics_fixed.npz')

# Save labels
np.savez(OUTPUT_DIR / 'labels.npz',
         teacher_labels=np.array(teacher_labels),
         student_labels=np.array(student_labels))

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
vmin = min(D_scaled.min(), D_fixed.min())
vmax = max(D_scaled.max(), D_fixed.max())

plot_heatmap(D_scaled, teacher_labels, student_labels,
             title='Procrustes residual — 15 teacher dirs × all students (free scale)',
             out_path=OUTPUT_DIR / 'heatmap_scaled.png', vmin=vmin, vmax=vmax)
plot_heatmap(D_fixed,  teacher_labels, student_labels,
             title='Procrustes residual — 15 teacher dirs × all students (fixed scale)',
             out_path=OUTPUT_DIR / 'heatmap_fixed.png',  vmin=vmin, vmax=vmax)
plot_comparison_heatmaps(D_scaled, D_fixed, teacher_labels, student_labels,
                         out_path=OUTPUT_DIR / 'heatmap_comparison.png')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '='*60)
print(f'Done.  Shape: {D_scaled.shape}')
print(f'Scaled — min={D_scaled.min():.4f}  max={D_scaled.max():.4f}  mean={D_scaled.mean():.4f}')
print(f'Fixed  — min={D_fixed.min():.4f}   max={D_fixed.max():.4f}   mean={D_fixed.mean():.4f}')
print(f'Output: {OUTPUT_DIR}')
