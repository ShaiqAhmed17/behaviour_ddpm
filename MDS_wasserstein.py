#!/usr/bin/env python3
"""
MDS of pairwise behaviour sliced-Wasserstein distances: all teachers and students.

Pipeline
--------
1. For each model (healthy teacher, 14 ablated teachers, all single-direction
   recovery students), run the sampler on N_TRIALS fixed trials and cache the
   output distributions to disk (restartable).
2. Compute the full N×N pairwise sliced-Wasserstein matrix from the cached
   samples (each checkpoint is loaded only once).
3. Fit metric and non-metric MDS on the distance matrix.
4. Visualise: teachers = circles, students = crosses, colour = ablation
   direction (shared between matched teacher/student pairs).

Output directory: results/wasserstein_mds/
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

# ─── Configuration ────────────────────────────────────────────────────────────

TEACHER_RUN   = 'index_cued_first_diffusion_0.3_swap_7'
SAMPLER_ROOT  = REPO_ROOT / 'results_link_sampler'
OUTPUT_DIR    = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'wasserstein_mds'
CACHE_DIR     = OUTPUT_DIR / 'cache'

N_TRIALS      = 16
N_SAMPLES     = 512
N_PROJECTIONS = 128
SEED          = 42
ANGLE_STEP    = 30

# ─── Trial helpers ────────────────────────────────────────────────────────────

def _generate_trials():
    angles = list(range(0, 360, ANGLE_STEP))
    return [
        {'cue': cue, 'color1': float(c1), 'color2': float(c2)}
        for cue in [1, 2]
        for c1 in angles
        for c2 in angles
    ]


def _select_trial_indices(n_total, n_select, seed):
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_total, size=n_select, replace=False))


def _build_trial_info(trial, task, device, num_samples):
    pf = torch.tensor([[trial['color1'], trial['color2']]]) * (np.pi / 180.0)
    rf = pf.clone()
    override = {'probe_features': pf, 'report_features': rf}
    override_cart = {
        f'{k}_cart': torch.stack(polar2cart(1.0, v), -1) for k, v in override.items()
    }
    override_dict = {
        **override, **override_cart,
        'cued_item_idx': torch.tensor([trial['cue'] - 1]),
    }
    var_dict = task.task_variable_gen.generate_variable_dict(
        batch_size=1, override_stimulus_features_dict=override_dict,
    )
    return task.generate_trial_information(
        batch_size=1, num_samples=num_samples,
        override_task_variable_information=var_dict,
    )


def _run_sampler(model, trial_info, device, num_samples, ablation_vector=None):
    prep = [x.to(device) if isinstance(x, torch.Tensor) else x
            for x in trial_info.prep_network_inputs]
    diff = [x.to(device) if isinstance(x, torch.Tensor) else x
            for x in trial_info.diffusion_network_inputs]
    kw = dict(
        prep_network_inputs=prep,
        diffusion_network_inputs=diff,
        prep_epoch_durations=trial_info.prep_epoch_durations,
        diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
        samples_shape=[1, num_samples],
        noise_scaler=1.0,
    )
    if ablation_vector is not None:
        kw['ablation_vector'] = ablation_vector
    with torch.no_grad():
        _, sd = model.generate_samples(**kw)
    s = sd['samples']
    return (s[0] if s.ndim == 3 else s).cpu().numpy().astype(np.float64)


def _load_ablation_vec(json_path, direction_idx, device):
    with open(json_path) as f:
        data = json.load(f)
    keys = sorted(data['nullspace_vectors']['vectors'].keys())
    v = np.array(data['nullspace_vectors']['vectors'][keys[direction_idx]], dtype=np.float32)
    t = torch.tensor(v, device=device)
    return t / (torch.norm(t) + 1e-12)


# ─── Model registry ───────────────────────────────────────────────────────────

def build_model_list():
    teacher_root = SAMPLER_ROOT / TEACHER_RUN
    nullspace_json = teacher_root / 'nullspace_and_projection.json'

    models = []

    # Healthy teacher (no ablation)
    models.append({
        'label':             'teacher_healthy',
        'role':              'teacher',
        'ablation_dir':      None,
        'args_path':         teacher_root / 'args.yaml',
        'checkpoint_path':   teacher_root / 'state.mdl',
        'nullspace_json':    None,
    })

    # Ablated teachers (dirs 0–13, same checkpoint)
    for di in range(14):
        models.append({
            'label':           f'teacher_dir{di:02d}',
            'role':            'teacher',
            'ablation_dir':    di,
            'args_path':       teacher_root / 'args.yaml',
            'checkpoint_path': teacher_root / 'state.mdl',
            'nullspace_json':  nullspace_json,
        })

    # Recovery students — single integer ablation directions only
    pattern = re.compile(
        r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(\d+)_(\d+)$'
    )
    for d in sorted(SAMPLER_ROOT.iterdir()):
        m = pattern.match(d.name)
        if not m:
            continue
        if not (d / 'state.mdl').exists() or not (d / 'args.yaml').exists():
            continue
        abl_dir  = int(m.group(1))
        seed_idx = int(m.group(2))
        models.append({
            'label':           f'student_dir{abl_dir:02d}_s{seed_idx}',
            'role':            'student',
            'ablation_dir':    abl_dir,
            'args_path':       d / 'args.yaml',
            'checkpoint_path': d / 'state.mdl',
            'nullspace_json':  None,
        })

    return models


# ─── Sample collection (checkpoint-batched, disk-cached) ──────────────────────

def collect_all_samples(models, all_trials, trial_indices, device):
    """
    Load each unique checkpoint once, generate samples for all associated models,
    cache to CACHE_DIR/<label>_samples.npy  shape (N_TRIALS, N_SAMPLES, 2).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    uncached = [m for m in models
                if not (CACHE_DIR / f"{m['label']}_samples.npy").exists()]

    if uncached:
        # Group uncached specs by checkpoint so we load each file only once
        by_ckpt = defaultdict(list)
        for m in uncached:
            by_ckpt[str(m['checkpoint_path'])].append(m)

        for ckpt_path_str, specs in by_ckpt.items():
            ckpt_path = Path(ckpt_path_str)
            print(f"\n  Loading checkpoint: {ckpt_path.parent.name}")
            _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
                str(specs[0]['args_path']), device
            )
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt)
            model.eval()

            for spec in specs:
                abl_vec = None
                if spec['nullspace_json'] is not None and spec['ablation_dir'] is not None:
                    abl_vec = _load_ablation_vec(
                        spec['nullspace_json'], spec['ablation_dir'], device
                    )

                trial_samples = []
                for idx_i, t_idx in enumerate(trial_indices):
                    ti = _build_trial_info(all_trials[t_idx], task, device, N_SAMPLES)
                    s  = _run_sampler(model, ti, device, N_SAMPLES, abl_vec)
                    trial_samples.append(s)

                arr = np.stack(trial_samples)          # (N_TRIALS, N_SAMPLES, 2)
                np.save(CACHE_DIR / f"{spec['label']}_samples.npy", arr)
                print(f"    cached {spec['label']}  {arr.shape}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        m['label']: np.load(CACHE_DIR / f"{m['label']}_samples.npy")
        for m in models
    }


# ─── Sliced-Wasserstein ───────────────────────────────────────────────────────

def _sw(X, Y, seed):
    rng  = np.random.default_rng(seed)
    dirs = rng.normal(size=(N_PROJECTIONS, X.shape[1])).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return float(np.mean([wasserstein_distance(X @ v, Y @ v) for v in dirs]))


def mean_sw_across_trials(sA, sB):
    """sA, sB: (N_TRIALS, N_SAMPLES, D).  Returns mean SW over trials."""
    return float(np.mean([_sw(sA[t], sB[t], SEED + t) for t in range(sA.shape[0])]))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    models = build_model_list()
    print(f"\n{len(models)} models: {sum(m['role']=='teacher' for m in models)} teachers, "
          f"{sum(m['role']=='student' for m in models)} students")
    for m in models:
        print(f"  [{m['role']:7s} dir={str(m['ablation_dir']):>4s}]  {m['label']}")

    all_trials    = _generate_trials()
    trial_indices = _select_trial_indices(len(all_trials), N_TRIALS, SEED)
    print(f"\nTrial indices ({N_TRIALS}): {trial_indices}")

    # ── 1. Sample ─────────────────────────────────────────────────────────────
    print("\n─── Step 1: Collecting behaviour samples ───")
    all_samples = collect_all_samples(models, all_trials, trial_indices, device)

    # ── 2. Pairwise Wasserstein ────────────────────────────────────────────────
    labels        = [m['label'] for m in models]
    N             = len(models)
    dist_npy      = OUTPUT_DIR / 'pairwise_wasserstein.npy'
    labels_npy    = OUTPUT_DIR / 'model_labels.npy'

    if dist_npy.exists():
        print("\n─── Step 2: Loading cached distance matrix ───")
        dist_matrix = np.load(dist_npy)
    else:
        print("\n─── Step 2: Computing pairwise Wasserstein ───")
        dist_matrix = np.zeros((N, N), dtype=np.float64)
        n_pairs = N * (N - 1) // 2
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = mean_sw_across_trials(all_samples[labels[i]], all_samples[labels[j]])
                dist_matrix[i, j] = dist_matrix[j, i] = d
                k += 1
                if k % 25 == 0 or k == n_pairs:
                    print(f"  {k}/{n_pairs}  {labels[i][:20]} | {labels[j][:20]}  d={d:.4f}")
        np.save(dist_npy,   dist_matrix)
        np.save(labels_npy, np.array(labels))
        print(f"  Saved: {dist_npy.name}")

    print(f"\n  Distance matrix: {dist_matrix.shape}  "
          f"range [{dist_matrix[dist_matrix>0].min():.4f}, {dist_matrix.max():.4f}]")

    # ── 3. MDS ────────────────────────────────────────────────────────────────
    print("\n─── Step 3: Fitting MDS ───")
    mds_m  = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    c_m    = mds_m.fit_transform(dist_matrix)
    print(f"  Metric    stress: {mds_m.stress_:.4f}")

    mds_nm = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42)
    c_nm   = mds_nm.fit_transform(dist_matrix)
    print(f"  Non-metric stress: {mds_nm.stress_:.4f}")

    # ── 4. Visualise ──────────────────────────────────────────────────────────
    print("\n─── Step 4: Visualising ───")

    # Colour palette: direction 0–13 get tab20; healthy teacher gets black
    tab20    = plt.get_cmap('tab20')
    dir_colors = {di: tab20(di / 14) for di in range(14)}

    def _color(m):
        if m['ablation_dir'] is None:
            return 'black'
        return dir_colors[m['ablation_dir']]

    def plot_mds(coords, title, fname):
        fig, ax = plt.subplots(figsize=(15, 12))

        for i, m in enumerate(models):
            x, y = coords[i]
            c = _color(m)
            if m['role'] == 'teacher':
                marker = '*' if m['ablation_dir'] is None else 'o'
                sz     = 600 if m['ablation_dir'] is None else 350
                ax.scatter(x, y, s=sz, marker=marker, c=[c],
                           edgecolors='black', linewidths=1.5, zorder=4)
                lbl = 'T★' if m['ablation_dir'] is None else f"T{m['ablation_dir']:02d}"
            else:
                ax.scatter(x, y, s=130, marker='X', c=[c],
                           edgecolors='black', linewidths=0.6, alpha=0.85, zorder=3)
                lbl = f"S{m['ablation_dir']:02d}"

            ax.annotate(
                lbl, (x, y),
                textcoords='offset points', xytext=(6, 4),
                fontsize=7, color=c,
                fontweight='bold' if m['role'] == 'teacher' else 'normal',
            )

        # Role legend
        role_handles = [
            mpatches.Patch(facecolor='grey', edgecolor='black',
                           label='Teacher (●, ★=healthy)'),
            mpatches.Patch(facecolor='grey', edgecolor='black',
                           label='Student (✕)', hatch='xx'),
        ]
        # Direction colour patches
        dir_handles = [
            mpatches.Patch(facecolor='black', edgecolor='black', label='healthy'),
        ] + [
            mpatches.Patch(facecolor=dir_colors[di], edgecolor='black',
                           label=f'dir {di:02d}')
            for di in range(14)
        ]

        leg1 = ax.legend(handles=role_handles, loc='upper left',  fontsize=9, title='Role')
        leg2 = ax.legend(handles=dir_handles,  loc='upper right', fontsize=8,
                         title='Ablation dir', ncol=2)
        ax.add_artist(leg1)

        ax.set_xlabel('MDS 1', fontsize=12)
        ax.set_ylabel('MDS 2', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fname}")

    plot_mds(c_m,
             f'Behaviour Wasserstein MDS (metric)  stress={mds_m.stress_:.3f}',
             'mds_wasserstein_metric.png')
    plot_mds(c_nm,
             f'Behaviour Wasserstein MDS (non-metric)  stress={mds_nm.stress_:.3f}',
             'mds_wasserstein_nonmetric.png')

    np.savez(
        OUTPUT_DIR / 'mds_wasserstein_data.npz',
        dist_matrix   = dist_matrix,
        labels        = np.array(labels),
        coords_metric = c_m,
        coords_nm     = c_nm,
        roles         = np.array([m['role'] for m in models]),
        ablation_dirs = np.array([
            -1 if m['ablation_dir'] is None else m['ablation_dir'] for m in models
        ]),
    )

    print(f"\n✓  Done.  {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
