#!/usr/bin/env python3
"""Compare teacher ablation directions against selected student runs.

This script reproduces the notebook comparison logic in a single process so it
can run unattended. It loads each student trajectory once, then compares every
teacher ablation direction against every student run and writes a CSV of
pairwise metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch

DEFAULT_STUDENT_RUNS = [
    'index_cued_first_diffusion_0.3_swap_recovery_ablation_0_0',
    'index_cued_first_diffusion_0.3_swap_recovery_ablation_1_0',
    'index_cued_first_diffusion_0.3_swap_recovery_ablation_4_0',
    'index_cued_first_diffusion_0.3_swap_recovery_ablation_5_0',
    'index_cued_first_diffusion_0.3_swap_recovery_ablation_6_0',
    'index_cued_first_diffusion_0.3_swap_ablation_7_1',
]


def _load_single_nullspace_vector(json_path, direction_idx, device):
    with open(json_path) as f:
        data = json.load(f)
    vectors = data['nullspace_vectors']['vectors']
    sorted_keys = sorted(vectors.keys())
    key = sorted_keys[int(direction_idx)]
    vec = np.array(vectors[key], dtype=np.float32)
    return torch.tensor(vec, device=device), key


def generate_trial_combinations(angle_step=30):
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for color1 in angles:
            for color2 in angles:
                trials.append(
                    {
                        'cue': cue,
                        'color1_angle': color1,
                        'color2_angle': color2,
                        'swap': False,
                    }
                )
    return trials


def _build_trial_info_and_inputs(trial, task, device, num_samples):
    probe_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180.0)
    report_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180.0)

    override_stimulus_features = {
        'probe_features': probe_features,
        'report_features': report_features,
    }
    override_stimulus_cart_features = {
        f'{k}_cart': torch.stack(polar2cart(1.0, v), -1)
        for k, v in override_stimulus_features.items()
    }
    override_stimulus_features_dict = {
        **override_stimulus_features,
        **override_stimulus_cart_features,
        'cued_item_idx': torch.tensor([trial['cue'] - 1]),
    }

    task_variable_dict = task.task_variable_gen.generate_variable_dict(
        batch_size=1,
        override_stimulus_features_dict=override_stimulus_features_dict,
    )
    trial_info = task.generate_trial_information(
        batch_size=1,
        num_samples=num_samples,
        override_task_variable_information=task_variable_dict,
    )

    prep_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in trial_info.prep_network_inputs]
    diff_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in trial_info.diffusion_network_inputs]
    return trial_info, prep_inputs, diff_inputs


def _extract_single_trial_prep_trajectory(
    trial,
    task,
    model,
    device,
    ablation_vector=None,
    neural_dim=16,
    prep_epoch_indices=None,
):
    with torch.no_grad():
        trial_info, prep_inputs, diff_inputs = _build_trial_info_and_inputs(
            trial,
            task,
            device=device,
            num_samples=1,
        )

        sample_kwargs = dict(
            prep_network_inputs=prep_inputs,
            diffusion_network_inputs=diff_inputs,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[1, 1],
            noise_scaler=1.0,
        )
        if ablation_vector is not None:
            sample_kwargs['ablation_vector'] = ablation_vector

        prep_dicts, _ = model.generate_samples(**sample_kwargs)

        if prep_epoch_indices is None:
            indices = range(len(prep_dicts))
        else:
            indices = prep_epoch_indices

        prep_segments = [
            prep_dicts[i]['preparatory_trajectory'][0, 0, :, :neural_dim].cpu().numpy().astype(np.float32)
            for i in indices
        ]
        return np.concatenate(prep_segments, axis=0)


def _extract_single_trial_behaviour_samples(
    trial,
    task,
    model,
    device,
    num_samples=512,
    ablation_vector=None,
):
    with torch.no_grad():
        trial_info, prep_inputs, diff_inputs = _build_trial_info_and_inputs(
            trial,
            task,
            device=device,
            num_samples=num_samples,
        )

        sample_kwargs = dict(
            prep_network_inputs=prep_inputs,
            diffusion_network_inputs=diff_inputs,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[1, num_samples],
            noise_scaler=1.0,
        )
        if ablation_vector is not None:
            sample_kwargs['ablation_vector'] = ablation_vector

        _, samples_dict = model.generate_samples(**sample_kwargs)
        samples = samples_dict['samples']
        # Expected shape is [1, num_samples, 2] for behaviour outputs.
        if samples.ndim == 3:
            out = samples[0]
        elif samples.ndim == 2:
            out = samples
        else:
            raise ValueError(f'Unexpected samples shape: {tuple(samples.shape)}')
        return out.cpu().numpy().astype(np.float64)


def _load_model_context(spec, device):
    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(str(spec['args_path']), device)
    ckpt = torch.load(spec['checkpoint_path'], map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    ablation_vector = None
    ablation_key = None
    if spec.get('nullspace_json_path') is not None and spec.get('ablation_direction_idx') is not None:
        ablation_vector, ablation_key = _load_single_nullspace_vector(
            spec['nullspace_json_path'],
            spec['ablation_direction_idx'],
            device,
        )

    return task, model, ablation_vector, ablation_key


def _extract_or_load_trajectories(spec, trials, device, neural_dim=16, prep_epoch_indices=None, trial_indices=None):
    source = spec['source']
    if source == 'array':
        arr_path = Path(spec['array_path'])
        if arr_path.suffix == '.npy':
            arr = np.load(arr_path)
        else:
            data = np.load(arr_path)
            if 'trajectories' in data:
                arr = data['trajectories']
            else:
                first_key = list(data.keys())[0]
                arr = data[first_key]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        return arr

    if source != 'checkpoint':
        raise ValueError(f"Unknown source '{source}'. Use 'checkpoint' or 'array'.")

    task, model, ablation_vector, ablation_key = _load_model_context(spec, device)
    if ablation_key is not None:
        print(f"[{spec['label']}] using ablation direction key: {ablation_key}")

    rows = []
    if trial_indices is None:
        iter_indices = list(range(len(trials)))
    else:
        iter_indices = [int(i) for i in trial_indices]

    for i, trial_idx in enumerate(iter_indices):
        if i % 50 == 0:
            print(f"[{spec['label']}] trial {i}/{len(iter_indices)}")
        trial = trials[int(trial_idx)]
        tr = _extract_single_trial_prep_trajectory(
            trial,
            task,
            model,
            device=device,
            ablation_vector=ablation_vector,
            neural_dim=neural_dim,
            prep_epoch_indices=prep_epoch_indices,
        )
        rows.append(tr)

    return np.stack(rows, axis=0).astype(np.float32)


def _extract_behaviour_samples_for_trials(spec, trials, trial_indices, device, num_samples=512):
    if spec['source'] != 'checkpoint':
        raise ValueError('Behaviour extraction currently supports source=checkpoint only.')

    task, model, ablation_vector, ablation_key = _load_model_context(spec, device)
    if ablation_key is not None:
        print(f"[{spec['label']}] using ablation direction key: {ablation_key}")

    behaviour = {}
    for idx in trial_indices:
        trial = trials[int(idx)]
        behaviour[int(idx)] = _extract_single_trial_behaviour_samples(
            trial,
            task,
            model,
            device=device,
            num_samples=num_samples,
            ablation_vector=ablation_vector,
        )
    return behaviour


def _select_trial_indices(n_total, n_select, seed):
    if n_select is None or n_select >= n_total:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(n_total), size=n_select, replace=False))
    return [int(i) for i in idx]


def _joint_pca_transform(X, Y, n_components):
    if n_components is None:
        return X, Y, None
    n_components = int(min(n_components, X.shape[1], Y.shape[1]))
    if n_components < 1:
        return X, Y, None
    pca = PCA(n_components=n_components)
    pca.fit(np.vstack([X, Y]))
    return pca.transform(X), pca.transform(Y), pca


def preprocess_average_trajectories(avg_A, avg_B, use_zscore=False, pca_dims=10):
    X = avg_A.astype(np.float64)
    Y = avg_B.astype(np.float64)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    if use_zscore:
        x_std = X.std(axis=0, keepdims=True) + 1e-8
        y_std = Y.std(axis=0, keepdims=True) + 1e-8
        X = X / x_std
        Y = Y / y_std

    return _joint_pca_transform(X, Y, pca_dims)


def _procrustes_residual(Xc, Yc_aligned):
    resid = Xc - Yc_aligned
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    rms_ref = float(np.sqrt(np.mean(Xc ** 2))) + 1e-8
    return rms_resid / rms_ref


def procrustes_distance(X, Y, allow_scaling=False):
    if X.shape != Y.shape:
        raise ValueError(f'Shape mismatch for Procrustes: {X.shape} vs {Y.shape}')

    if allow_scaling:
        m1, m2, disparity = procrustes(X, Y)
        return {
            'distance': float(disparity),
            'aligned_ref': m1,
            'aligned_cmp': m2,
            'rotation': None,
            'mode': 'full_procrustes_with_scaling',
        }

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    R, _ = orthogonal_procrustes(Yc, Xc)
    Y_aligned = Yc @ R

    resid = Xc - Y_aligned
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    rms_ref = float(np.sqrt(np.mean(Xc ** 2))) + 1e-8
    dist = rms_resid / rms_ref

    return {
        'distance': float(dist),
        'aligned_ref': Xc,
        'aligned_cmp': Y_aligned,
        'rotation': R,
        'mode': 'orthogonal_no_scaling',
    }


def procrustes_curve(traj_A, traj_B, pca_dims=10):
    """Time-resolved Procrustes: for each timepoint t, compare the n_trials×neural_dim
    snapshot matrices under (a) an independent per-timepoint rotation Q_t and (b) a
    single shared rotation Q fit on the mean trajectories.

    Returns
    -------
    dict with:
        'curve_independent' : (T,) float array — residual at each t under Q_t
        'curve_shared_q'    : (T,) float array — residual at each t under fixed Q
        'auc_independent'   : scalar AUC of curve_independent
        'auc_shared_q'      : scalar AUC of curve_shared_q
        'shared_Q'          : (d, d) rotation matrix
        'pca'               : fitted PCA (or None if pca_dims is None)
    """
    n_trials, T, d = traj_A.shape
    assert traj_B.shape == (n_trials, T, d), f"Shape mismatch: {traj_A.shape} vs {traj_B.shape}"

    # Optionally reduce dimension via joint PCA fitted on mean trajectories.
    avg_A = traj_A.mean(axis=0).astype(np.float64)  # (T, d)
    avg_B = traj_B.mean(axis=0).astype(np.float64)
    if pca_dims is not None and pca_dims < d:
        pca = PCA(n_components=int(pca_dims))
        pca.fit(np.vstack([avg_A, avg_B]))
        traj_A_r = pca.transform(traj_A.reshape(-1, d)).reshape(n_trials, T, pca_dims).astype(np.float64)
        traj_B_r = pca.transform(traj_B.reshape(-1, d)).reshape(n_trials, T, pca_dims).astype(np.float64)
        avg_A_r = pca.transform(avg_A)
        avg_B_r = pca.transform(avg_B)
        d_r = pca_dims
    else:
        pca = None
        traj_A_r, traj_B_r = traj_A.astype(np.float64), traj_B.astype(np.float64)
        avg_A_r, avg_B_r = avg_A, avg_B
        d_r = d

    # Shared Q: fit one rotation on the mean trajectories (all timepoints stacked).
    mean_Ac = avg_A_r - avg_A_r.mean(axis=0, keepdims=True)
    mean_Bc = avg_B_r - avg_B_r.mean(axis=0, keepdims=True)
    shared_Q, _ = orthogonal_procrustes(mean_Bc, mean_Ac)

    curve_indep = np.empty(T, dtype=np.float64)
    curve_shared = np.empty(T, dtype=np.float64)

    for t in range(T):
        Xt = traj_A_r[:, t, :]  # (n_trials, d_r)
        Yt = traj_B_r[:, t, :]
        Xtc = Xt - Xt.mean(axis=0, keepdims=True)
        Ytc = Yt - Yt.mean(axis=0, keepdims=True)

        # Independent Q_t
        Qt, _ = orthogonal_procrustes(Ytc, Xtc)
        curve_indep[t] = _procrustes_residual(Xtc, Ytc @ Qt)

        # Shared Q
        curve_shared[t] = _procrustes_residual(Xtc, Ytc @ shared_Q)

    dt = 1.0 / T
    return {
        'curve_independent': curve_indep,
        'curve_shared_q': curve_shared,
        'auc_independent': float(np.trapz(curve_indep) * dt),
        'auc_shared_q': float(np.trapz(curve_shared) * dt),
        'shared_Q': shared_Q,
        'pca': pca,
    }


def centroid_distance_pointcloud(cloud_A, cloud_B):
    cA = cloud_A.mean(axis=0)
    cB = cloud_B.mean(axis=0)
    return float(np.linalg.norm(cA - cB))


def sliced_wasserstein_distance(X, Y, n_projections=128, seed=42):
    rng = np.random.default_rng(seed)
    dim = X.shape[1]
    dirs = rng.normal(size=(n_projections, dim)).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8

    vals = []
    for v in dirs:
        vals.append(wasserstein_distance(X @ v, Y @ v))
    return float(np.mean(vals))


def rbf_mmd2(X, Y, gamma=None):
    if gamma is None:
        Z = np.vstack([X, Y])
        med = np.median(np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2))
        gamma = 1.0 / (2.0 * (med ** 2 + 1e-8))

    XX = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    YY = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    XY = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))

    return float(XX.mean() + YY.mean() - 2.0 * XY.mean())


def _compute_metric_bundle(traj_A, traj_B, label_A, label_B, seed=42, use_zscore=False, pca_dims=10, allow_scaling=False, n_sw_projections=128):
    if traj_A.shape[1] != traj_B.shape[1]:
        raise ValueError(f"Trajectory length mismatch: {traj_A.shape[1]} vs {traj_B.shape[1]}")

    pc = procrustes_curve(traj_A, traj_B, pca_dims=pca_dims)

    cloud_A = traj_A.reshape(-1, traj_A.shape[-1]).astype(np.float64)
    cloud_B = traj_B.reshape(-1, traj_B.shape[-1]).astype(np.float64)

    return {
        'model_A': label_A,
        'model_B': label_B,
        'n_trials': int(traj_A.shape[0]),
        'prep_steps': int(traj_A.shape[1]),
        'neural_dim': int(traj_A.shape[2]),
        'procrustes_auc': pc['auc_independent'],
        'procrustes_auc_shared_q': pc['auc_shared_q'],
        # Legacy scalar on mean trajectory for backward compatibility.
        'procrustes_distance': pc['auc_independent'],
        'centroid_distance_pointcloud': float(centroid_distance_pointcloud(cloud_A, cloud_B)),
        # Retained for diagnostic comparison only; not the training-objective metric.
        'sliced_wasserstein_neural': float(sliced_wasserstein_distance(cloud_A, cloud_B, n_projections=n_sw_projections, seed=seed)),
        'mmd2_rbf': float(rbf_mmd2(cloud_A, cloud_B)),
        'procrustes_curve_independent': pc['curve_independent'].tolist(),
        'procrustes_curve_shared_q': pc['curve_shared_q'].tolist(),
    }


def _compute_behaviour_wasserstein_bundle(behaviour_A, behaviour_B, trial_indices, n_sw_projections=128, seed=42):
    per_trial = []
    for i, t_idx in enumerate(trial_indices):
        XA = behaviour_A[int(t_idx)]
        XB = behaviour_B[int(t_idx)]
        wd = sliced_wasserstein_distance(XA, XB, n_projections=n_sw_projections, seed=seed + i)
        per_trial.append((int(t_idx), float(wd)))

    vals = np.array([v for _, v in per_trial], dtype=np.float64)
    return {
        'sliced_wasserstein': float(vals.mean()),
        'sliced_wasserstein_behaviour_mean': float(vals.mean()),
        'sliced_wasserstein_behaviour_median': float(np.median(vals)),
        'sliced_wasserstein_behaviour_std': float(vals.std()),
        'sliced_wasserstein_behaviour_max': float(vals.max()),
        'behaviour_trials_evaluated': int(len(per_trial)),
    }, per_trial


def _style_label(run_name):
    return run_name.replace('index_cued_first_diffusion_0.3_swap_', '')


def generate_pretty_heatmaps(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['student_short'] = df['student_run'].map(_style_label)
    df['ablation_short'] = [f"dir_{int(i):02d}" for i in df['ablation_direction_idx'].values]

    abl_order = [f'dir_{i:02d}' for i in sorted(df['ablation_direction_idx'].unique())]
    stu_order = sorted(df['student_short'].unique(), key=lambda s: (
        [int(c) if c.isdigit() else c for c in __import__('re').split(r'(\d+)', s)]
    ))

    proc_col = 'procrustes_auc' if 'procrustes_auc' in df.columns else 'procrustes_distance'

    # Exclude recovery_ablation_6 from neural panel (anomalous scale).
    neural_stu_order = [s for s in stu_order if 'recovery_ablation_6' not in s]
    proc_mat = df.pivot(index='ablation_short', columns='student_short', values=proc_col).reindex(index=abl_order, columns=neural_stu_order)

    # Exclude ablation direction 11 from behaviour panel.
    was_abl_order = [a for a in abl_order if a != 'dir_11']
    was_mat = df.pivot(index='ablation_short', columns='student_short', values='sliced_wasserstein').reindex(index=was_abl_order, columns=stu_order)

    sns.set_theme(style='whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5), gridspec_kw={'width_ratios': [1, 1]})

    hm0 = sns.heatmap(
        proc_mat,
        ax=axes[0],
        cmap='rocket',
        annot=True,
        fmt='.3f',
        linewidths=0.6,
        linecolor='white',
        cbar_kws={'label': 'Procrustes AUC (Neural)' if proc_col == 'procrustes_auc' else 'Procrustes Distance (Neural)'},
    )
    axes[0].set_title('Neural Geometry Similarity', fontsize=15, pad=12)
    axes[0].set_xlabel('Student Run', fontsize=12)
    axes[0].set_ylabel('Teacher Ablation Direction', fontsize=12)
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].tick_params(axis='y', rotation=0)
    hm0.collections[0].colorbar.ax.tick_params(labelsize=10)

    hm1 = sns.heatmap(
        was_mat,
        ax=axes[1],
        cmap='mako_r',
        annot=True,
        fmt='.2f',
        linewidths=0.6,
        linecolor='white',
        cbar_kws={'label': 'Sliced Wasserstein (Behaviour 2D)'},
    )
    axes[1].set_title('Behaviour Distribution Similarity', fontsize=15, pad=12)
    axes[1].set_xlabel('Student Run', fontsize=12)
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].tick_params(axis='y', rotation=0)
    hm1.collections[0].colorbar.ax.tick_params(labelsize=10)

    fig.suptitle('Teacher Ablations vs Students: Neural and Behaviour Metrics', fontsize=17, y=1.02)
    fig.tight_layout()

    out_png = output_dir / 'heatmap_neural_procrustes_and_behaviour_wasserstein.png'
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')


def compare_ablation_dirs_to_students(
    teacher_root_dir,
    student_runs,
    device,
    direction_indices=None,
    angle_step=30,
    neural_dim=16,
    prep_epoch_indices=None,
    use_zscore=False,
    pca_dims=10,
    allow_scaling=False,
    n_sw_projections=128,
    num_behaviour_samples=512,
    wasserstein_trials=16,
    procrustes_trials=None,
    seed=42,
    output_dir=None,
):
    if direction_indices is None:
        direction_indices = list(range(14))

    teacher_root = REPO_ROOT / 'results_link_sampler' / teacher_root_dir
    args_path = teacher_root / 'args.yaml'
    ckpt_path = teacher_root / 'state.mdl'
    nullspace_json_path = teacher_root / 'nullspace_and_projection.json'

    trials = generate_trial_combinations(angle_step=angle_step)
    wasserstein_trial_indices = _select_trial_indices(len(trials), wasserstein_trials, seed)
    if procrustes_trials is None:
        procrustes_trial_indices = list(wasserstein_trial_indices)
    else:
        procrustes_trial_indices = _select_trial_indices(len(trials), procrustes_trials, seed + 13)

    print(f'Using {len(procrustes_trial_indices)} matched trials for neural Procrustes: {procrustes_trial_indices}')
    print(f'Using {len(wasserstein_trial_indices)} matched trials for behaviour Wasserstein: {wasserstein_trial_indices}')

    student_trajs = {}
    student_behaviour = {}
    for s in student_runs:
        print(f'Loading student run: {s}')
        sroot = REPO_ROOT / 'results_link_sampler' / s
        spec = {
            'label': s,
            'source': 'checkpoint',
            'args_path': sroot / 'args.yaml',
            'checkpoint_path': sroot / 'state.mdl',
            'nullspace_json_path': None,
            'ablation_direction_idx': None,
            'array_path': None,
        }
        student_trajs[s] = _extract_or_load_trajectories(
            spec,
            trials,
            device=device,
            neural_dim=neural_dim,
            prep_epoch_indices=prep_epoch_indices,
            trial_indices=procrustes_trial_indices,
        )
        student_behaviour[s] = _extract_behaviour_samples_for_trials(
            spec,
            trials,
            trial_indices=wasserstein_trial_indices,
            device=device,
            num_samples=num_behaviour_samples,
        )

    rows = []
    per_trial_rows = []
    for di in direction_indices:
        print('\n' + '=' * 70)
        print(f'Processing ablation direction {di}')
        ablated_spec = {
            'label': f'{teacher_root_dir}_ablated_dir_{int(di):02d}',
            'source': 'checkpoint',
            'args_path': args_path,
            'checkpoint_path': ckpt_path,
            'nullspace_json_path': nullspace_json_path,
            'ablation_direction_idx': int(di),
            'array_path': None,
        }
        ablated_traj = _extract_or_load_trajectories(
            ablated_spec,
            trials,
            device=device,
            neural_dim=neural_dim,
            prep_epoch_indices=prep_epoch_indices,
            trial_indices=procrustes_trial_indices,
        )
        ablated_behaviour = _extract_behaviour_samples_for_trials(
            ablated_spec,
            trials,
            trial_indices=wasserstein_trial_indices,
            device=device,
            num_samples=num_behaviour_samples,
        )

        for s, st_traj in student_trajs.items():
            print(f'  Comparing ablation {di} -> student {s}')
            bundle = _compute_metric_bundle(
                ablated_traj,
                st_traj,
                label_A=ablated_spec['label'],
                label_B=s,
                seed=seed + int(di),
                use_zscore=use_zscore,
                pca_dims=pca_dims,
                allow_scaling=allow_scaling,
                n_sw_projections=n_sw_projections,
            )
            behaviour_bundle, per_trial = _compute_behaviour_wasserstein_bundle(
                ablated_behaviour,
                student_behaviour[s],
                trial_indices=wasserstein_trial_indices,
                n_sw_projections=n_sw_projections,
                seed=seed + int(di),
            )
            bundle.update(behaviour_bundle)
            bundle['ablation_direction_idx'] = int(di)
            bundle['student_run'] = s
            bundle['num_behaviour_samples'] = int(num_behaviour_samples)
            rows.append(bundle)

            for t_idx, val in per_trial:
                per_trial_rows.append(
                    {
                        'ablation_direction_idx': int(di),
                        'student_run': s,
                        'trial_index': int(t_idx),
                        'sliced_wasserstein_behaviour': float(val),
                        'num_behaviour_samples': int(num_behaviour_samples),
                    }
                )

    df = pd.DataFrame(rows)
    if output_dir is None:
        output_dir = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'ablations_vs_students'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / 'ablations_vs_students_scores.csv'
    out_trials_csv = output_dir / 'ablations_vs_students_wasserstein_per_trial.csv'
    df.to_csv(out_csv, index=False)
    pd.DataFrame(per_trial_rows).to_csv(out_trials_csv, index=False)
    print(f'Saved: {out_csv}')
    print(f'Saved: {out_trials_csv}')
    generate_pretty_heatmaps(df, output_dir)
    return df, out_csv


def parse_args():
    parser = argparse.ArgumentParser(description='Compare teacher ablations to selected student runs.')
    parser.add_argument('--teacher-root-dir', default='index_cued_first_diffusion_0.3_swap_7')
    parser.add_argument('--student-runs', nargs='*', default=DEFAULT_STUDENT_RUNS)
    parser.add_argument('--direction-indices', nargs='*', type=int, default=None)
    parser.add_argument('--angle-step', type=int, default=30)
    parser.add_argument('--neural-dim', type=int, default=16)
    parser.add_argument('--prep-epoch-indices', nargs='*', type=int, default=None)
    parser.add_argument('--use-zscore', action='store_true')
    parser.add_argument('--pca-dims', type=int, default=10)
    parser.add_argument('--allow-scaling', action='store_true')
    parser.add_argument('--n-sw-projections', type=int, default=128)
    parser.add_argument('--num-behaviour-samples', type=int, default=512)
    parser.add_argument('--wasserstein-trials', type=int, default=16)
    parser.add_argument('--procrustes-trials', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    compare_ablation_dirs_to_students(
        teacher_root_dir=args.teacher_root_dir,
        student_runs=args.student_runs,
        device=device,
        direction_indices=args.direction_indices,
        angle_step=args.angle_step,
        neural_dim=args.neural_dim,
        prep_epoch_indices=args.prep_epoch_indices,
        use_zscore=args.use_zscore,
        pca_dims=args.pca_dims,
        allow_scaling=args.allow_scaling,
        n_sw_projections=args.n_sw_projections,
        num_behaviour_samples=args.num_behaviour_samples,
        wasserstein_trials=args.wasserstein_trials,
        procrustes_trials=args.procrustes_trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()