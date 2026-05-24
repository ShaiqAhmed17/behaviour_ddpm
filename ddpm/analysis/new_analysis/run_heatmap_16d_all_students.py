"""
Run the full teacher × student Procrustes heatmaps in 16-D for all recovery ablation students.

Teachers : abl00 – abl13  (ablated nullspace direction 0-13)
Students : recovery_ablation_0_0, recovery_ablation_1_0, recovery_ablation_4_0,
           recovery_ablation_5_0, recovery_ablation_6_0, recovery_ablation_no_ablation_0

Results saved to results/procrustes_heatmap/ with student tag in filename.
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from ddpm.analysis.new_analysis.procrustes_heatmap import compute_heatmap

CACHE_DIR  = os.path.join(_HERE, 'results', 'procrustes_heatmap', 'traj_cache')
OUTPUT_DIR = os.path.join(_HERE, 'results', 'procrustes_heatmap')
PREFIX     = 'index_cued_first_diffusion_0.3_swap_7'

TEACHER_TAGS = [f'abl{i:02d}' for i in range(14)]   # abl00 … abl13
STUDENT_TAGS = [
    'recovery_ablation_0_0',
    'recovery_ablation_1_0',
    'recovery_ablation_4_0',
    'recovery_ablation_5_0',
    'recovery_ablation_6_0',
    'recovery_ablation_no_ablation_0',
]

S = 512

# ---------------------------------------------------------------------------
# Load trajectories
# ---------------------------------------------------------------------------
def load(tag):
    if 'recovery' in tag or 'ablation' in tag:
        fname = f'index_cued_first_diffusion_0.3_swap_{tag}_unablated_S{S}.npy'
    else:
        fname = f'{PREFIX}_{tag}_S{S}.npy'
    path = os.path.join(CACHE_DIR, fname)
    arr = np.load(path).astype(np.float32)
    print(f'  loaded {os.path.basename(path)}  {arr.shape}')
    return arr

print('Loading teacher trajectories …')
teacher_trajs = [load(t) for t in TEACHER_TAGS]
for tag, arr in zip(TEACHER_TAGS, teacher_trajs):
    assert arr.shape[-1] == 16, f'{tag}: expected 16-D, got {arr.shape}'

# ---------------------------------------------------------------------------
# Run heatmap for each student
# ---------------------------------------------------------------------------
for student_tag in STUDENT_TAGS:
    print(f'\n{"="*70}')
    print(f'Student: {student_tag}')
    print(f'{"="*70}')

    print('Loading student trajectories …')
    student_trajs = [load(student_tag)]
    assert student_trajs[0].shape[-1] == 16, f'{student_tag}: expected 16-D'

    print(f'\nRunning compute_heatmap  ({len(teacher_trajs)} teachers × {len(student_trajs)} student) …')
    result = compute_heatmap(
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

    # Save
    out_scaled = os.path.join(OUTPUT_DIR, f'heatmap_procrustes_scaled_16d_{student_tag}.npy')
    np.save(out_scaled, result['residuals'])
    print(f'\nSaved residuals  → {out_scaled}')
    print('Residuals (14 teachers × 1 student):')
    print(result['residuals'].flatten())

    # Save diagnostics
    n_t = len(teacher_trajs)
    n_s = len(student_trajs)
    identity_residuals = np.array([
        [result['alignment_results'][i][j].identity_residual for j in range(n_s)]
        for i in range(n_t)
    ])
    scales = np.array([
        [result['alignment_results'][i][j].c for j in range(n_s)]
        for i in range(n_t)
    ])

    diag_path = os.path.join(OUTPUT_DIR, f'diagnostics_scaled_16d_{student_tag}.npz')
    np.savez(
        diag_path,
        residuals=result['residuals'],
        identity_residuals=identity_residuals,
        scales=scales,
        teacher_tags=np.array(TEACHER_TAGS),
        student_tag=np.array([student_tag]),
    )
    print(f'Saved diagnostics → {diag_path}')

print(f'\n{"="*70}')
print('All heatmaps complete!')
print(f'{"="*70}')
