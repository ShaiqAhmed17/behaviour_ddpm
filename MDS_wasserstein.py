#!/usr/bin/env python3
"""
MDS of pairwise behaviour sliced-Wasserstein distances: all teachers and students.

Pipeline
--------
1. For each model (healthy teacher, 14 ablated teachers, all recovery students
   including single-dir, multi-dir, PCA-ablation, and no-ablation variants),
   run the sampler on N_TRIALS fixed trials and cache the output distributions
   to disk (restartable).
2. Compute the full N×N pairwise sliced-Wasserstein matrix, extending an
   existing matrix if present (only new pairs are computed).
3. Fit metric and non-metric MDS on the distance matrix.
4. Visualise: teachers = stars, students = markers by type, colour = ablation
   direction (shared between matched teacher/student pairs).

Student types and markers:
  single  – single nullspace direction (D, diamond)
  multi   – multiple nullspace directions (s, square)
  pca     – PCA-based ablation direction (^, triangle)
  none    – no ablation (P, plus-filled)

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
from matplotlib.lines import Line2D
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart
from ddpm.utils.vis.style import set_publication_style, save_figure, save_legend

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

# Models excluded because they are outliers in the MDS embedding
_EXCLUDE_LABELS = {
    'teacher_dir03',      # y = +3.4 in metric MDS
    'teacher_dir00',      # x = +1.5, y = -2.0
    'student_dir00_s1',   # x = +1.3, y = -1.7
    'student_multi_0-1_s0',  # x = +2.3
}


def build_model_list():
    teacher_root = SAMPLER_ROOT / TEACHER_RUN
    nullspace_json = teacher_root / 'nullspace_and_projection.json'

    models = []

    # Healthy teacher (no ablation)
    models.append({
        'label':        'teacher_healthy',
        'role':         'teacher',
        'student_type': None,
        'ablation_dir': None,
        'args_path':    teacher_root / 'args.yaml',
        'checkpoint_path': teacher_root / 'state.mdl',
        'nullspace_json':  None,
    })

    # Ablated teachers; dirs 6, 8, 9, 10, 11 excluded (no students / outliers)
    for di in range(14):
        if di in (6, 8, 9, 10, 11):
            continue
        models.append({
            'label':        f'teacher_dir{di:02d}',
            'role':         'teacher',
            'student_type': None,
            'ablation_dir': di,
            'args_path':    teacher_root / 'args.yaml',
            'checkpoint_path': teacher_root / 'state.mdl',
            'nullspace_json':  nullspace_json,
        })

    # Regex patterns for student discovery
    # Single-dir nullspace: ...ablation_0_0, ...ablation_13_0, etc.
    pat_single = re.compile(
        r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(\d+)_(\d+)$'
    )
    # Multi-dir nullspace: ...ablation_0-1_0, ...ablation_0-1-2_0, etc.
    pat_multi = re.compile(
        r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(\d+(?:-\d+)+)_(\d+)$'
    )
    # PCA ablation (single or multi dir): ...pca_ablation_0_0, ...pca_ablation_0-1_0, etc.
    pat_pca = re.compile(
        r'^index_cued_first_diffusion_0\.3_swap_recovery_pca_ablation_(\d+(?:-\d+)*)_(\d+)$'
    )
    # No-ablation recovery: ...recovery_no_ablation_0
    pat_noabl = re.compile(
        r'^index_cued_first_diffusion_0\.3_swap_recovery_no_ablation_(\d+)$'
    )
    # No-ablation recovery (ablation-pipeline variant): ...recovery_ablation_no_ablation_0
    pat_noabl_abl = re.compile(
        r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_no_ablation_(\d+)$'
    )

    for d in sorted(SAMPLER_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not (d / 'state.mdl').exists() or not (d / 'args.yaml').exists():
            continue

        # Single-dir (try first — must not contain a dash in the dir field)
        m = pat_single.match(d.name)
        if m:
            abl_dir  = int(m.group(1))
            seed_idx = int(m.group(2))
            if abl_dir in (9, 10, 11):
                continue
            label = f'student_dir{abl_dir:02d}_s{seed_idx}'
            if label == 'student_dir06_s0':
                continue
            models.append({
                'label':        label,
                'role':         'student',
                'student_type': 'single',
                'ablation_dir': abl_dir,
                'args_path':    d / 'args.yaml',
                'checkpoint_path': d / 'state.mdl',
                'nullspace_json':  None,
            })
            continue

        # Multi-dir nullspace (compound dirs with dashes)
        m = pat_multi.match(d.name)
        if m:
            dirs_str = m.group(1)
            seed_idx = int(m.group(2))
            first_dir = int(dirs_str.split('-')[0])
            if first_dir in (9, 10, 11):
                continue
            models.append({
                'label':        f'student_multi_{dirs_str}_s{seed_idx}',
                'role':         'student',
                'student_type': 'multi',
                'ablation_dir': first_dir,
                'args_path':    d / 'args.yaml',
                'checkpoint_path': d / 'state.mdl',
                'nullspace_json':  None,
            })
            continue

        # PCA ablation
        m = pat_pca.match(d.name)
        if m:
            dirs_str = m.group(1)
            seed_idx = int(m.group(2))
            first_dir = int(dirs_str.split('-')[0])
            if first_dir in (9, 10, 11):
                continue
            models.append({
                'label':        f'student_pca_{dirs_str}_s{seed_idx}',
                'role':         'student',
                'student_type': 'pca',
                'ablation_dir': first_dir,
                'args_path':    d / 'args.yaml',
                'checkpoint_path': d / 'state.mdl',
                'nullspace_json':  None,
            })
            continue

        # No-ablation recovery
        m = pat_noabl.match(d.name)
        if m:
            seed_idx = int(m.group(1))
            models.append({
                'label':        f'student_noabl_s{seed_idx}',
                'role':         'student',
                'student_type': 'none',
                'ablation_dir': None,
                'args_path':    d / 'args.yaml',
                'checkpoint_path': d / 'state.mdl',
                'nullspace_json':  None,
            })
            continue

        # No-ablation recovery (ablation-pipeline variant)
        m = pat_noabl_abl.match(d.name)
        if m:
            seed_idx = int(m.group(1))
            models.append({
                'label':        f'student_noabl_abl_s{seed_idx}',
                'role':         'student',
                'student_type': 'none',
                'ablation_dir': None,
                'args_path':    d / 'args.yaml',
                'checkpoint_path': d / 'state.mdl',
                'nullspace_json':  None,
            })
            continue

    return [m for m in models if m['label'] not in _EXCLUDE_LABELS]


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


# ─── Distance matrix (with incremental extension) ─────────────────────────────

def build_distance_matrix(models, all_samples, dist_npy, labels_npy):
    labels = [m['label'] for m in models]
    N = len(models)

    if dist_npy.exists() and labels_npy.exists():
        old_labels = np.load(labels_npy, allow_pickle=True).tolist()
        old_matrix = np.load(dist_npy)
        old_idx    = {l: i for i, l in enumerate(old_labels)}

        new_labels = [l for l in labels if l not in old_idx]

        if not new_labels:
            print("─── Step 2: Distance matrix up to date — loading ───")
            # Reorder to match current model list (handles additions/reorderings)
            dist_matrix = np.zeros((N, N), dtype=np.float64)
            for i, li in enumerate(labels):
                for j, lj in enumerate(labels):
                    if li in old_idx and lj in old_idx:
                        dist_matrix[i, j] = old_matrix[old_idx[li], old_idx[lj]]
            return dist_matrix

        print(f"─── Step 2: Extending matrix with {len(new_labels)} new model(s) ───")
        print(f"  New: {new_labels}")

        new_label_set = set(new_labels)
        dist_matrix   = np.zeros((N, N), dtype=np.float64)

        # Copy existing pairs
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if li in old_idx and lj in old_idx:
                    dist_matrix[i, j] = old_matrix[old_idx[li], old_idx[lj]]

        # Compute only pairs involving at least one new model
        new_pairs = [
            (i, j)
            for i in range(N) for j in range(i + 1, N)
            if labels[i] in new_label_set or labels[j] in new_label_set
        ]
        print(f"  Computing {len(new_pairs)} new pairs ...")
        for k, (i, j) in enumerate(new_pairs):
            d = mean_sw_across_trials(all_samples[labels[i]], all_samples[labels[j]])
            dist_matrix[i, j] = dist_matrix[j, i] = d
            if (k + 1) % 10 == 0 or k + 1 == len(new_pairs):
                print(f"  {k+1}/{len(new_pairs)}  {labels[i][:25]} | {labels[j][:25]}  d={d:.4f}")

    else:
        print("─── Step 2: Computing pairwise Wasserstein from scratch ───")
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

    np.save(dist_npy,    dist_matrix)
    np.save(labels_npy,  np.array(labels))
    print(f"  Saved: {dist_npy.name}")
    return dist_matrix


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    models = build_model_list()
    n_teachers = sum(m['role'] == 'teacher' for m in models)
    n_students = sum(m['role'] == 'student' for m in models)
    print(f"\n{len(models)} models: {n_teachers} teachers, {n_students} students")
    for m in models:
        stype = m.get('student_type') or ''
        print(f"  [{m['role']:7s} {stype:6s} dir={str(m['ablation_dir']):>4s}]  {m['label']}")

    all_trials    = _generate_trials()
    trial_indices = _select_trial_indices(len(all_trials), N_TRIALS, SEED)
    print(f"\nTrial indices ({N_TRIALS}): {trial_indices}")

    # ── 1. Sample ─────────────────────────────────────────────────────────────
    print("\n─── Step 1: Collecting behaviour samples ───")
    all_samples = collect_all_samples(models, all_trials, trial_indices, device)

    # ── 2. Pairwise Wasserstein ────────────────────────────────────────────────
    labels   = [m['label'] for m in models]
    dist_npy = OUTPUT_DIR / 'pairwise_wasserstein.npy'
    labels_npy = OUTPUT_DIR / 'model_labels.npy'
    dist_matrix = build_distance_matrix(models, all_samples, dist_npy, labels_npy)

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

    tab20      = plt.get_cmap('tab20')
    dir_colors = {di: tab20(di / 14) for di in range(14)}

    # Marker per student_type
    _MARKERS = {
        None:     '*',   # teacher
        'single': 'D',
        'multi':  's',
        'pca':    '^',
        'none':   'P',
    }

    def _color(m):
        if m['ablation_dir'] is None:
            return 'black'
        return dir_colors[int(m['ablation_dir'])]

    def _marker(m):
        if m['role'] == 'teacher':
            return '*'
        return _MARKERS.get(m.get('student_type'), 'D')

    def plot_mds(coords, title, fname):
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, m in enumerate(models):
            x, y = coords[i]
            c = _color(m)
            mk = _marker(m)
            if m['role'] == 'teacher':
                ax.scatter(x, y, s=400, marker=mk, c=[c],
                           edgecolors='black', linewidths=1.5, zorder=4)
            else:
                ax.scatter(x, y, s=130, marker=mk, c=[c],
                           edgecolors='black', linewidths=0.6, alpha=0.85, zorder=3)

        # Tight limits with 8% padding around the data
        xs, ys = coords[:, 0], coords[:, 1]
        xpad = 0.08 * (xs.max() - xs.min() or 1)
        ypad = 0.08 * (ys.max() - ys.min() or 1)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

        ax.set_xlabel('MDS 1 (a.u.)', fontsize=18)
        ax.set_ylabel('MDS 2 (a.u.)', fontsize=18)
        ax.set_title(title, fontweight='bold', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.25)
        fig.tight_layout()

        out_stem = OUTPUT_DIR / fname.replace('.png', '')
        save_figure(fig, out_stem)

        # Standalone legends only — not added to main plot
        role_handles = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
                   markeredgecolor='black', markersize=14, label='Teacher'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='grey',
                   markeredgecolor='black', markersize=9,  label='Student (single dir)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',
                   markeredgecolor='black', markersize=9,  label='Student (multi dir)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='grey',
                   markeredgecolor='black', markersize=9,  label='Student (PCA dir)'),
            Line2D([0], [0], marker='P', color='w', markerfacecolor='grey',
                   markeredgecolor='black', markersize=9,  label='Student (no ablation)'),
        ]
        dir_handles = [
            mpatches.Patch(facecolor='black', edgecolor='black', label='healthy / no abl'),
        ] + [
            mpatches.Patch(facecolor=dir_colors[di], edgecolor='black', label=f'dir {di:02d}')
            for di in range(14) if di not in (9, 10, 11)
        ]
        save_legend(role_handles, [h.get_label() for h in role_handles],
                    out_stem.parent / (out_stem.name + '_role'))
        save_legend(dir_handles, [h.get_label() for h in dir_handles],
                    out_stem.parent / (out_stem.name + '_dir'), ncol=2)
        plt.close(fig)
        print(f"  Saved: {out_stem}.pdf / .png + legends")

    plot_mds(c_m,  'Wasserstein MDS (metric)',     'mds_wasserstein_metric.png')
    plot_mds(c_nm, 'Wasserstein MDS (non-metric)', 'mds_wasserstein_nonmetric.png')

    np.savez(
        OUTPUT_DIR / 'mds_wasserstein_data.npz',
        dist_matrix   = dist_matrix,
        labels        = np.array(labels),
        coords_metric = c_m,
        coords_nm     = c_nm,
        roles         = np.array([m['role'] for m in models]),
        student_types = np.array([m.get('student_type') or '' for m in models]),
        ablation_dirs = np.array([
            -1 if m['ablation_dir'] is None else m['ablation_dir'] for m in models
        ]),
    )

    print(f"\n✓  Done.  {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
