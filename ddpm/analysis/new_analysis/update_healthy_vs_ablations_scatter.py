#!/usr/bin/env python3
"""Recompute healthy vs ablated-teacher scatter plot using updated metrics.

Updated metrics:
  - Procrustes  : procrustes_curve AUC  (time-resolved, per-trial snapshots)
  - Wasserstein : behaviour sliced-SW   (on diffusion output distributions)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

TEACHER_RUN = 'index_cued_first_diffusion_0.3_swap_7'
N_DIRECTIONS = 14
ANGLE_STEP = 30
NEURAL_DIM = 16
N_SW_PROJECTIONS = 128
NUM_BEHAVIOUR_SAMPLES = 512
WASSERSTEIN_TRIALS = 16
PCA_DIMS = 10
SEED = 42

OUTPUT_DIR = REPO_ROOT / 'ddpm/analysis/new_analysis/results/healthy_vs_ablations'


def generate_trials(angle_step=30):
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for c1 in angles:
            for c2 in angles:
                trials.append({'cue': cue, 'color1_angle': c1, 'color2_angle': c2, 'swap': False})
    return trials


def _build_inputs(trial, task, device, num_samples):
    probe_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180.0)
    report_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180.0)
    override = {
        'probe_features': probe_features,
        'report_features': report_features,
    }
    override_cart = {f'{k}_cart': torch.stack(polar2cart(1.0, v), -1) for k, v in override.items()}
    override.update(override_cart)
    override['cued_item_idx'] = torch.tensor([trial['cue'] - 1])
    tv = task.task_variable_gen.generate_variable_dict(batch_size=1, override_stimulus_features_dict=override)
    ti = task.generate_trial_information(batch_size=1, num_samples=num_samples, override_task_variable_information=tv)
    prep = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ti.prep_network_inputs]
    diff = [x.to(device) if isinstance(x, torch.Tensor) else x for x in ti.diffusion_network_inputs]
    return ti, prep, diff


def extract_prep_trajectory(trial, task, model, device, ablation_vector=None):
    with torch.no_grad():
        ti, prep, diff = _build_inputs(trial, task, device, num_samples=1)
        kwargs = dict(
            prep_network_inputs=prep,
            diffusion_network_inputs=diff,
            prep_epoch_durations=ti.prep_epoch_durations,
            diffusion_epoch_durations=ti.diffusion_epoch_durations,
            samples_shape=[1, 1],
            noise_scaler=1.0,
        )
        if ablation_vector is not None:
            kwargs['ablation_vector'] = ablation_vector
        prep_dicts, _ = model.generate_samples(**kwargs)
        segments = [prep_dicts[i]['preparatory_trajectory'][0, 0, :, :NEURAL_DIM].cpu().numpy()
                    for i in range(len(prep_dicts))]
        return np.concatenate(segments, axis=0).astype(np.float32)


def extract_behaviour_samples(trial, task, model, device, num_samples, ablation_vector=None):
    with torch.no_grad():
        ti, prep, diff = _build_inputs(trial, task, device, num_samples=num_samples)
        kwargs = dict(
            prep_network_inputs=prep,
            diffusion_network_inputs=diff,
            prep_epoch_durations=ti.prep_epoch_durations,
            diffusion_epoch_durations=ti.diffusion_epoch_durations,
            samples_shape=[1, num_samples],
            noise_scaler=1.0,
        )
        if ablation_vector is not None:
            kwargs['ablation_vector'] = ablation_vector
        _, samples_dict = model.generate_samples(**kwargs)
        s = samples_dict['samples']
        return (s[0] if s.ndim == 3 else s).cpu().numpy().astype(np.float64)


def load_model(run_dir, device):
    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(run_dir / 'args.yaml'), device
    )
    ckpt = torch.load(run_dir / 'state.mdl', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return task, model


def load_nullspace_vector(json_path, direction_idx, device):
    with open(json_path) as f:
        data = json.load(f)
    vectors = data['nullspace_vectors']['vectors']
    key = sorted(vectors.keys())[int(direction_idx)]
    vec = np.array(vectors[key], dtype=np.float32)
    return torch.tensor(vec, device=device)


def _procrustes_residual(Xc, Yc_aligned):
    resid = Xc - Yc_aligned
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    rms_ref = float(np.sqrt(np.mean(Xc ** 2))) + 1e-8
    return rms_resid / rms_ref


def procrustes_curve_auc(traj_A, traj_B, pca_dims=10):
    """Time-resolved Procrustes AUC on per-trial snapshot matrices."""
    n_trials, T, d = traj_A.shape
    avg_A = traj_A.mean(axis=0).astype(np.float64)
    avg_B = traj_B.mean(axis=0).astype(np.float64)
    if pca_dims is not None and pca_dims < d:
        pca = PCA(n_components=int(pca_dims))
        pca.fit(np.vstack([avg_A, avg_B]))
        traj_A_r = pca.transform(traj_A.reshape(-1, d)).reshape(n_trials, T, pca_dims).astype(np.float64)
        traj_B_r = pca.transform(traj_B.reshape(-1, d)).reshape(n_trials, T, pca_dims).astype(np.float64)
    else:
        traj_A_r = traj_A.astype(np.float64)
        traj_B_r = traj_B.astype(np.float64)

    curve = np.empty(T, dtype=np.float64)
    for t in range(T):
        Xtc = traj_A_r[:, t, :] - traj_A_r[:, t, :].mean(axis=0, keepdims=True)
        Ytc = traj_B_r[:, t, :] - traj_B_r[:, t, :].mean(axis=0, keepdims=True)
        Qt, _ = orthogonal_procrustes(Ytc, Xtc)
        curve[t] = _procrustes_residual(Xtc, Ytc @ Qt)

    return float(np.trapezoid(curve) / T)


def sliced_wasserstein(X, Y, n_projections=128, seed=42):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    dirs = rng.normal(size=(n_projections, d)).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return float(np.mean([wasserstein_distance(X @ v, Y @ v) for v in dirs]))


def behaviour_sliced_wasserstein(beh_A, beh_B, trial_indices, n_projections=128, seed=42):
    vals = [sliced_wasserstein(beh_A[t], beh_B[t], n_projections=n_projections, seed=seed + i)
            for i, t in enumerate(trial_indices)]
    return float(np.mean(vals))


def collect_data(task, model, trials, trial_indices_proc, trial_indices_was, device, ablation_vector=None, label=''):
    # Prep trajectories for Procrustes
    trajs = []
    for i, idx in enumerate(trial_indices_proc):
        if i % 5 == 0:
            print(f'  [{label}] prep trajectory {i+1}/{len(trial_indices_proc)}')
        trajs.append(extract_prep_trajectory(trials[idx], task, model, device, ablation_vector))
    traj_arr = np.stack(trajs, axis=0)  # [n_proc, T, D]

    # Behaviour samples for Wasserstein
    behaviour = {}
    for i, idx in enumerate(trial_indices_was):
        print(f'  [{label}] behaviour sample {i+1}/{len(trial_indices_was)}')
        behaviour[idx] = extract_behaviour_samples(trials[idx], task, model, device, NUM_BEHAVIOUR_SAMPLES, ablation_vector)

    return traj_arr, behaviour


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    teacher_root = REPO_ROOT / 'results_link_sampler' / TEACHER_RUN
    nullspace_json = teacher_root / 'nullspace_and_projection.json'

    trials = generate_trials(ANGLE_STEP)
    print(f'Generated {len(trials)} trials.')

    rng = np.random.default_rng(SEED)
    wasserstein_indices = sorted(rng.choice(len(trials), size=WASSERSTEIN_TRIALS, replace=False).tolist())
    procrustes_indices = wasserstein_indices  # use same subset
    print(f'Trial indices: {wasserstein_indices}')

    print('\n=== Loading healthy teacher ===')
    task, model = load_model(teacher_root, device)
    healthy_traj, healthy_beh = collect_data(
        task, model, trials, procrustes_indices, wasserstein_indices, device,
        ablation_vector=None, label='healthy'
    )

    rows = []
    for di in range(N_DIRECTIONS):
        print(f'\n=== Ablation direction {di} ===')
        ablation_vector = load_nullspace_vector(nullspace_json, di, device)

        ablated_traj, ablated_beh = collect_data(
            task, model, trials, procrustes_indices, wasserstein_indices, device,
            ablation_vector=ablation_vector, label=f'dir_{di:02d}'
        )

        proc_auc = procrustes_curve_auc(healthy_traj, ablated_traj, pca_dims=PCA_DIMS)
        beh_sw = behaviour_sliced_wasserstein(
            healthy_beh, ablated_beh, wasserstein_indices,
            n_projections=N_SW_PROJECTIONS, seed=SEED + di
        )

        print(f'  procrustes_auc={proc_auc:.4f}  behaviour_sw={beh_sw:.4f}')
        rows.append({
            'ablation_direction_idx': di,
            'procrustes_auc': proc_auc,
            'sliced_wasserstein_behaviour': beh_sw,
        })

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / 'healthy_vs_teacher_ablations_updated_scores.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved scores: {csv_path}')

    _make_scatter(df)


def _scatter_panel(ax, df, title):
    sc = ax.scatter(
        df['procrustes_auc'],
        df['sliced_wasserstein_behaviour'],
        c=df['ablation_direction_idx'],
        cmap='tab20',
        vmin=0,
        vmax=13,
        s=100,
        edgecolors='k',
        linewidths=0.6,
        zorder=3,
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"{int(row['ablation_direction_idx'])}",
            (row['procrustes_auc'], row['sliced_wasserstein_behaviour']),
            textcoords='offset points',
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel('Procrustes AUC (time-resolved, per-trial)', fontsize=11)
    ax.set_ylabel('Sliced Wasserstein (behaviour)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    return sc


def _make_scatter(df):
    df_no11 = df[df['ablation_direction_idx'] != 11].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = _scatter_panel(axes[0], df, 'All directions (dir 11 is outlier)')
    _scatter_panel(axes[1], df_no11, 'Excluding dir 11')

    fig.colorbar(sc, ax=axes, pad=0.02, label='Ablation direction index')
    fig.suptitle('Healthy teacher vs ablated teacher — updated metrics\n'
                 '(Procrustes AUC × behaviour sliced Wasserstein)',
                 fontsize=12)
    fig.tight_layout()

    out_png = OUTPUT_DIR / 'procrustes_vs_sliced_wasserstein_updated.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {out_png}')


if __name__ == '__main__':
    main()
