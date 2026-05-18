"""
Compute geometric metrics directly from pre-sampled ablated teacher trajectories.

Scans `results_link_sampler/*/ablated_teacher_trajectories` for sweep files,
loads postprep states for a selected prep index, and computes per-run metrics:
 - mean radius (mean L2 norm)
 - primary PCA explained variance (PC1)
 - cue separation (L2 between cue1 and cue2 mean states)
 - plane angle between cue subspaces (principal angle)

Writes CSV to `ddpm/analysis/new_analysis/results/ablation_teacher_metrics.csv`.

Usage:
    python3 ddpm/analysis/new_analysis/compute_ablation_teacher_metrics.py
"""
import torch
import numpy as np
from pathlib import Path
import json
import csv
from sklearn.decomposition import PCA

REPO = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_ROOT = REPO / 'results_link_sampler'
OUT_DIR = REPO / 'ddpm' / 'analysis' / 'new_analysis' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'ablation_teacher_metrics.csv'


def list_ablated_dirs(results_root):
    dirs = []
    for p in sorted(results_root.iterdir()):
        traj_dir = p / 'ablated_teacher_trajectories'
        if traj_dir.exists() and traj_dir.is_dir():
            dirs.append((p.name, traj_dir))
    return dirs


def principal_angle_between_subspaces(A, B, k=2):
    # A, B: arrays [n_samples, dim]
    # compute top-k left singular vectors (basis) via PCA
    pcaA = PCA(n_components=k).fit(A)
    pcaB = PCA(n_components=k).fit(B)
    UA = pcaA.components_.T  # dim x k
    UB = pcaB.components_.T
    # compute singular values of UA^T UB
    M = UA.T.dot(UB)
    try:
        s = np.linalg.svd(M, compute_uv=False)
    except Exception:
        s = np.linalg.svd(M + 1e-12 * np.eye(M.shape[0]), compute_uv=False)
    # clamp s to [-1,1]
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    # return largest principal angle in degrees
    return float(np.degrees(np.max(angles)))


def process_traj_dir(traj_dir, prep_idx=2, max_files=20):
    files = sorted(traj_dir.glob('ablated_teacher_trajectories_sweep_step_*.pt'))
    if not files:
        return None
    files = files[:max_files]

    states_all = []
    cues_all = []
    for f in files:
        data = torch.load(str(f), map_location='cpu', weights_only=False)
        trials = data.get('sweep_batch_trials', None)
        prep_dicts = data.get('ablated_teacher_prep_dicts', None)
        if trials is None or prep_dicts is None:
            continue
        if prep_idx not in prep_dicts:
            continue
        postprep = prep_dicts[prep_idx]['postprep_state']  # [batch, samples, dim]
        # reduce samples by mean
        postprep_reduced = postprep.mean(dim=1).numpy()
        # collect trials
        for i, tr in enumerate(trials):
            cue = int(tr.get('cue', 1))
            states_all.append(postprep_reduced[i])
            cues_all.append(cue)

    if not states_all:
        return None

    states = np.vstack(states_all)  # [N, dim]
    cues = np.array(cues_all)

    # metrics
    radius_mean = float(np.mean(np.linalg.norm(states, axis=1)))
    pca = PCA(n_components=min(10, states.shape[1]))
    pca.fit(states)
    pca_primary = float(pca.explained_variance_ratio_[0])

    # cue separation
    mask1 = cues == 1
    mask2 = cues == 2
    if mask1.sum() == 0 or mask2.sum() == 0:
        cue_sep = float(0.0)
        plane_angle = float(0.0)
    else:
        mean1 = states[mask1].mean(axis=0)
        mean2 = states[mask2].mean(axis=0)
        cue_sep = float(np.linalg.norm(mean1 - mean2))
        # plane angle between cue subspaces
        try:
            plane_angle = principal_angle_between_subspaces(states[mask1], states[mask2], k=2)
        except Exception:
            plane_angle = float(0.0)

    return {
        'n_samples': int(states.shape[0]),
        'dim': int(states.shape[1]),
        'radius_mean': radius_mean,
        'pca_primary_var': pca_primary,
        'cue_separation': cue_sep,
        'plane_angle': plane_angle,
    }


def main():
    dirs = list_ablated_dirs(RESULTS_ROOT)
    if not dirs:
        print('No ablated_teacher_trajectories directories found under', RESULTS_ROOT)
        return 1

    rows = []
    for run_name, traj_dir in dirs:
        print('Processing', run_name)
        metrics = process_traj_dir(traj_dir)
        if metrics is None:
            print('  skipped', run_name)
            continue
        row = {'run_name': run_name, 'traj_dir': str(traj_dir)}
        row.update(metrics)
        rows.append(row)

    # write CSV
    keys = ['run_name', 'traj_dir', 'n_samples', 'dim', 'radius_mean', 'pca_primary_var', 'cue_separation', 'plane_angle']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('Wrote', OUT_CSV)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
