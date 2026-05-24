"""
Run the full teacher × student Procrustes heatmap in 16-D (full hidden state).

Teachers : abl00 – abl13  (ablated nullspace direction 0-13)
Student  : swap_recovery_8_unablated

No nullspace projection is applied — trajectories are used as-is (16-D).
Results are saved to results/procrustes_heatmap/ with a _16d suffix.
"""

import os
import sys
import numpy as np

# Ensure repo root is on the path when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from ddpm.analysis.new_analysis.procrustes_heatmap import compute_heatmap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR  = os.path.join(_HERE, 'results', 'procrustes_heatmap', 'traj_cache')
OUTPUT_DIR = os.path.join(_HERE, 'results', 'procrustes_heatmap')
PREFIX     = 'index_cued_first_diffusion_0.3_swap_7'

TEACHER_TAGS = [f'abl{i:02d}' for i in range(14)]   # abl00 … abl13
STUDENT_TAGS = ['swap_recovery_8_unablated']

S = 512

# ---------------------------------------------------------------------------
# Load trajectories
# ---------------------------------------------------------------------------
def load(tag):
    if 'recovery' in tag or 'ablation' in tag:
        # student files have a different prefix pattern
        fname = f'index_cued_first_diffusion_0.3_{tag}_S{S}.npy'
    else:
        fname = f'{PREFIX}_{tag}_S{S}.npy'
    path = os.path.join(CACHE_DIR, fname)
    arr = np.load(path).astype(np.float32)
    print(f'  loaded {os.path.basename(path)}  {arr.shape}')
    return arr

print('Loading teacher trajectories …')
teacher_trajs = [load(t) for t in TEACHER_TAGS]

print('Loading student trajectories …')
student_trajs = [load(t) for t in STUDENT_TAGS]

# Sanity-check: all must be 16-D
for tag, arr in zip(TEACHER_TAGS + STUDENT_TAGS, teacher_trajs + student_trajs):
    assert arr.shape[-1] == 16, f'{tag}: expected 16-D, got {arr.shape}'

# ---------------------------------------------------------------------------
# Run heatmap (scaled Procrustes)
# ---------------------------------------------------------------------------
print(f'\nRunning compute_heatmap  ({len(teacher_trajs)} teachers × {len(student_trajs)} students) …')
result_scaled = compute_heatmap(
    teacher_trajs,
    student_trajs,
    allow_scaling=True,
    n_restarts=3,
    max_iter=50,
    tol=1e-5,
    n_jobs_lap=-1,
    seed=42,
    device='cuda',
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_scaled = os.path.join(OUTPUT_DIR, 'heatmap_procrustes_scaled_16d.npy')
np.save(out_scaled, result_scaled['residuals'])
print(f'\nSaved residuals  → {out_scaled}')
print('Residuals (14 teachers × 1 student):')
print(result_scaled['residuals'])

# Save full diagnostics (alignment results are not pickleable directly, so save
# the per-cell residuals, identity_residuals, and scale factors as arrays)
n_t = len(teacher_trajs)
n_s = len(student_trajs)
identity_residuals = np.array([
    [result_scaled['alignment_results'][i][j].identity_residual for j in range(n_s)]
    for i in range(n_t)
])
scales = np.array([
    [result_scaled['alignment_results'][i][j].c for j in range(n_s)]
    for i in range(n_t)
])

diag_path = os.path.join(OUTPUT_DIR, 'diagnostics_scaled_16d.npz')
np.savez(
    diag_path,
    residuals=result_scaled['residuals'],
    identity_residuals=identity_residuals,
    scales=scales,
    teacher_tags=np.array(TEACHER_TAGS),
    student_tags=np.array(STUDENT_TAGS),
)
print(f'Saved diagnostics → {diag_path}')
