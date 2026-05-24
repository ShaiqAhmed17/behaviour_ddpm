#!/usr/bin/env python3
"""
run_procrustes_heatmap.py

Driver script for the teacher-student Procrustes heatmap analysis.

Workflow
--------
1. Load (or extract) preparatory trajectories for every teacher and student.
2. Project to the teacher's 14-D behavioural nullspace.
3. Run compute_heatmap twice: once with free scaling (c free) and once with
   c=1 (pure orthogonal Procrustes).
4. Save residual matrices + per-cell diagnostics as .npy / .npz.
5. Run permutation tests on both matrices.
6. Produce heatmap visualisations and the permutation-test histograms.

Usage
-----
python run_procrustes_heatmap.py \\
    --teacher-run  index_cued_first_diffusion_0.3_swap_7 \\
    --student-runs index_cued_first_diffusion_0.3_swap_recovery_ablation_0_0 \\
                   index_cued_first_diffusion_0.3_swap_recovery_ablation_1_0 \\
    --n-samples 8 \\
    --output-dir results/procrustes_heatmap

Pass --include-unablated-teacher to add the unablated teacher as an extra row.
Pass --direction-indices 0 1 2 ... to restrict ablation directions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from ddpm.analysis.new_analysis.procrustes_heatmap import (
    project_to_nullspace,
    align_trajectories,
    compute_heatmap,
    permutation_test,
)


# ---------------------------------------------------------------------------
# Trial generation (identical to run_ablation_student_comparison.py)
# ---------------------------------------------------------------------------

def _generate_trials(angle_step: int = 30) -> list:
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for a1 in angles:
            for a2 in angles:
                trials.append({'cue': cue, 'color1_angle': a1, 'color2_angle': a2})
    return trials


def _build_trial_network_inputs(trial: dict, task, device: str, num_samples: int):
    probe_features  = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180.0)
    report_features = probe_features.clone()
    override = {
        'probe_features': probe_features,
        'report_features': report_features,
    }
    override_cart = {
        f'{k}_cart': torch.stack(polar2cart(1.0, v), -1) for k, v in override.items()
    }
    full_override = {
        **override, **override_cart,
        'cued_item_idx': torch.tensor([trial['cue'] - 1]),
    }
    task_var = task.task_variable_gen.generate_variable_dict(
        batch_size=1, override_stimulus_features_dict=full_override
    )
    trial_info = task.generate_trial_information(
        batch_size=1, num_samples=num_samples,
        override_task_variable_information=task_var,
    )
    prep_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                   for x in trial_info.prep_network_inputs]
    diff_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                   for x in trial_info.diffusion_network_inputs]
    return trial_info, prep_inputs, diff_inputs


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

def _extract_trajectories(
    trials: list,
    task,
    model,
    device: str,
    n_samples: int,
    ablation_vector=None,
    trial_indices=None,
    chunk_size: int = 64,
    include_diffusion: bool = False,
) -> np.ndarray:
    """
    Extract trajectories for a subset of trials.

    Returns ndarray of shape (N_trials, n_samples, T, 16) where T is:
      - T_prep (sum of prep epoch durations) when include_diffusion=False
      - T_prep + T_diff                       when include_diffusion=True

    The S=n_samples independent trajectories arise from different random
    initial states drawn inside model.prepare().

    chunk_size controls how many samples are generated per model call;
    results are concatenated across chunks.  Use a smaller value if GPU
    memory is tight with large n_samples.
    """
    if trial_indices is None:
        trial_indices = list(range(len(trials)))

    rows = []
    with torch.no_grad():
        for idx_pos, trial_idx in enumerate(trial_indices):
            if idx_pos % 50 == 0:
                print(f'  extracting trial {idx_pos}/{len(trial_indices)} ...', flush=True)
            trial = trials[trial_idx]

            # Accumulate chunks along the sample dimension.
            chunk_epoch_lists: list[list[np.ndarray]] = []  # [chunk][epoch] -> (S_chunk, T, 16)
            remaining = n_samples
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
                remaining -= this_chunk

                trial_info, prep_inputs, diff_inputs = _build_trial_network_inputs(
                    trial, task, device, num_samples=this_chunk
                )
                sample_kwargs = dict(
                    prep_network_inputs=prep_inputs,
                    diffusion_network_inputs=diff_inputs,
                    prep_epoch_durations=trial_info.prep_epoch_durations,
                    diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
                    samples_shape=[1, this_chunk],
                    noise_scaler=1.0,
                )
                if ablation_vector is not None:
                    sample_kwargs['ablation_vector'] = ablation_vector

                prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)
                # prep_dicts[epoch]['preparatory_trajectory']: (1, this_chunk, T_epoch, 16)
                chunk_epochs = [
                    pd['preparatory_trajectory'][0].cpu().numpy()  # (this_chunk, T_epoch, 16)
                    for pd in prep_dicts
                ]
                if include_diffusion:
                    # embedded_sample_trajectory: (1, this_chunk, T_diff, 16) — ambient 16D path
                    chunk_epochs.append(
                        samples_dict['embedded_sample_trajectory'][0].cpu().numpy()
                    )
                chunk_epoch_lists.append(chunk_epochs)

            # Concatenate chunks then epochs.
            n_epochs = len(chunk_epoch_lists[0])
            epoch_trajs = [
                np.concatenate([chunk[ep] for chunk in chunk_epoch_lists], axis=0)  # (S, T_epoch, 16)
                for ep in range(n_epochs)
            ]
            full_traj = np.concatenate(epoch_trajs, axis=1)    # (S, T_total, 16)
            rows.append(full_traj)

    return np.stack(rows, axis=0).astype(np.float32)           # (N, S, T, 16)


def _load_model(run_dir: Path, device: str):
    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(run_dir / 'args.yaml'), device
    )
    ckpt = torch.load(run_dir / 'state.mdl', map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return task, model


def _load_nullspace(run_dir: Path) -> np.ndarray:
    """Load the 14 nullspace basis vectors from the teacher's JSON file."""
    with open(run_dir / 'nullspace_and_projection.json') as f:
        data = json.load(f)
    vecs = data['nullspace_vectors']['vectors']
    keys = sorted(vecs.keys())
    return np.stack([np.array(vecs[k], dtype=np.float64) for k in keys])  # (14, 16)


def _get_ablation_vector(model, direction_idx: int):
    vec = model.behaviour_nullspace[direction_idx].clone()
    vec = vec / (vec.norm() + 1e-12)
    return vec


# ---------------------------------------------------------------------------
# Trajectory loading (with optional caching)
# ---------------------------------------------------------------------------

def load_or_extract_trajectories(
    run_dir: Path,
    trials: list,
    device: str,
    n_samples: int,
    ablation_direction_idx: int | None,
    trial_indices: list | None,
    cache_dir: Path | None,
    label: str,
    chunk_size: int = 64,
    include_diffusion: bool = False,
) -> np.ndarray:
    """
    Load from cache if available, otherwise extract from model checkpoint.
    Returns (N, S, T, 16) float32.
      include_diffusion=False: T = T_prep  (prep epochs only, legacy behaviour)
      include_diffusion=True:  T = T_prep + T_diff (full path through 16D space)
    chunk_size: max samples per model.generate_samples call (prevents OOM for large S).
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        abl_part = f'abl{ablation_direction_idx:02d}' if ablation_direction_idx is not None else 'unablated'
        diff_part = '_full' if include_diffusion else ''
        cache_path = cache_dir / f'{label}_{abl_part}{diff_part}_S{n_samples}.npy'
        if cache_path.exists():
            print(f'  [cache] loading {cache_path}')
            return np.load(cache_path)

    print(f'  extracting {label} (ablation={ablation_direction_idx}, include_diffusion={include_diffusion}) ...', flush=True)
    task, model = _load_model(run_dir, device)

    ablation_vector = None
    if ablation_direction_idx is not None:
        ablation_vector = _get_ablation_vector(model, ablation_direction_idx)

    traj = _extract_trajectories(
        trials, task, model, device, n_samples,
        ablation_vector=ablation_vector,
        trial_indices=trial_indices,
        chunk_size=chunk_size,
        include_diffusion=include_diffusion,
    )

    if cache_dir is not None:
        np.save(cache_path, traj)
        print(f'  [cache] saved {cache_path}')

    del model, task
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return traj


# ---------------------------------------------------------------------------
# Saving diagnostics
# ---------------------------------------------------------------------------

def save_heatmap_diagnostics(heatmap_result: dict, out_path: Path) -> None:
    """
    Save per-cell diagnostics from compute_heatmap to a .npz file.

    Stored arrays:
        residuals          : (n_teachers, n_students)
        identity_residuals : (n_teachers, n_students)
        c_values           : (n_teachers, n_students)
        restart_residuals  : (n_teachers, n_students, n_restarts)
        trace_lengths      : (n_teachers, n_students)  — number of iterations
    """
    res_matrix = heatmap_result['residuals']
    all_results = heatmap_result['alignment_results']
    n_t, n_s = res_matrix.shape
    n_restarts = len(all_results[0][0].restart_residuals)

    id_res = np.empty((n_t, n_s))
    c_vals = np.empty((n_t, n_s))
    restart_res = np.empty((n_t, n_s, n_restarts))
    trace_len = np.empty((n_t, n_s), dtype=np.int32)

    for i in range(n_t):
        for j in range(n_s):
            r = all_results[i][j]
            id_res[i, j]       = r.identity_residual
            c_vals[i, j]       = r.c
            restart_res[i, j]  = r.restart_residuals
            trace_len[i, j]    = len(r.objective_trace)

    np.savez(
        out_path,
        residuals=res_matrix,
        identity_residuals=id_res,
        c_values=c_vals,
        restart_residuals=restart_res,
        trace_lengths=trace_len,
    )
    print(f'Saved diagnostics: {out_path}')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    D: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    n_t, n_s = D.shape
    fig_w = max(7, 0.65 * n_s + 2)
    fig_h = max(5, 0.65 * n_t + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(D, aspect='auto', cmap='viridis_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Procrustes residual')

    ax.set_xticks(range(n_s))
    ax.set_xticklabels(col_labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Student', fontsize=11)
    ax.set_ylabel('Teacher', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    for i in range(n_t):
        for j in range(n_s):
            ax.text(j, i, f'{D[i, j]:.3f}', ha='center', va='center',
                    fontsize=7, color='white' if D[i, j] < (D.max() + D.min()) / 2 else 'black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved heatmap: {out_path}')


def plot_permutation_test(perm_result: dict, title: str, out_path: Path) -> None:
    null = perm_result['null_distribution']
    obs  = perm_result['observed_T']
    pval = perm_result['p_value']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=60, density=True, color='steelblue', alpha=0.7, label='Null distribution')
    ax.axvline(obs, color='crimson', linewidth=2,
               label=f'Observed T = {obs:.4f}\np = {pval:.4f}')
    ax.set_xlabel('T = mean(diag) − mean(off-diag)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved permutation test plot: {out_path}')


def plot_comparison_heatmaps(
    D_scaled: np.ndarray,
    D_fixed: np.ndarray,
    row_labels: list,
    col_labels: list,
    out_path: Path,
) -> None:
    """Side-by-side: scaled vs fixed-c heatmaps on a shared colour scale."""
    vmin = min(D_scaled.min(), D_fixed.min())
    vmax = max(D_scaled.max(), D_fixed.max())
    n_t, n_s = D_scaled.shape
    cell_size = 0.65

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * (cell_size * n_s + 2), cell_size * n_t + 2.5),
        sharey=True,
    )
    for ax, D, subtitle in zip(axes, [D_scaled, D_fixed], ['Free scale (c optimised)', 'Fixed scale (c = 1)']):
        im = ax.imshow(D, aspect='auto', cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_s))
        ax.set_xticklabels(col_labels, rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel('Student', fontsize=10)
        ax.set_title(subtitle, fontsize=11)
        for i in range(n_t):
            for j in range(n_s):
                ax.text(j, i, f'{D[i, j]:.3f}', ha='center', va='center',
                        fontsize=6, color='white' if D[i, j] < (vmax + vmin) / 2 else 'black')

    axes[0].set_ylabel('Teacher', fontsize=10)
    fig.suptitle('Procrustes residual: robustness to scaling assumption', fontsize=12, y=1.01)
    fig.colorbar(im, ax=axes, label='Procrustes residual', shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved comparison heatmap: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Procrustes heatmap of teacher-student trajectory alignment')
    p.add_argument('--teacher-run',  default='index_cued_first_diffusion_0.3_swap_7')
    p.add_argument('--student-runs', nargs='+', default=[
        'index_cued_first_diffusion_0.3_swap_recovery_ablation_0_0',
        'index_cued_first_diffusion_0.3_swap_recovery_ablation_1_0',
        'index_cued_first_diffusion_0.3_swap_recovery_ablation_4_0',
        'index_cued_first_diffusion_0.3_swap_recovery_ablation_5_0',
        'index_cued_first_diffusion_0.3_swap_recovery_ablation_6_0',
        'index_cued_first_diffusion_0.3_swap_ablation_7_1',
    ])
    p.add_argument('--direction-indices', nargs='*', type=int, default=None,
                   help='Ablation direction indices (default: 0..13)')
    p.add_argument('--include-unablated-teacher', action='store_true',
                   help='Add the unablated teacher as an extra heatmap row')
    p.add_argument('--include-unablated-student', action='store_true',
                   help='Add the unablated teacher model as an extra column')
    p.add_argument('--include-diffusion', action='store_true',
                   help='Append the 40-step denoising trajectory after prep epochs, giving '
                        'a full (T_prep + T_diff)-step path through 16D space. '
                        'Cache files are written with a _full suffix.')
    p.add_argument('--angle-step', type=int, default=30,
                   help='Degrees between colour angles for trial generation')
    p.add_argument('--n-trials',   type=int, default=None,
                   help='Randomly sub-select this many trials (default: all)')
    p.add_argument('--n-samples',  type=int, default=8,
                   help='Independent denoising samples per trial (S dimension)')
    p.add_argument('--n-restarts', type=int, default=3)
    p.add_argument('--max-iter',   type=int, default=50)
    p.add_argument('--tol',        type=float, default=1e-5)
    p.add_argument('--n-jobs',     type=int, default=-1,
                   help='joblib threads for LAP step (-1 = all CPUs)')
    p.add_argument('--n-permutations', type=int, default=10_000)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu',
                   help='Device for Procrustes computation (cuda / cpu)')
    p.add_argument('--extraction-device', default=None,
                   help='Device for model inference / trajectory extraction (default: same as --device)')
    p.add_argument('--sample-chunk-size', type=int, default=64,
                   help='Max samples per generate_samples call; reduce if GPU OOMs during extraction')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--cache-dir',  default=None,
                   help='Directory for caching extracted trajectory arrays')
    p.add_argument('--results-root', default=str(REPO_ROOT / 'results_link_sampler'))
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    extraction_device = args.extraction_device if args.extraction_device else device
    rng = np.random.default_rng(args.seed)

    results_root = Path(args.results_root)
    teacher_run_dir = results_root / args.teacher_run

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'procrustes_heatmap'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (output_dir / 'traj_cache')

    # ------------------------------------------------------------------
    # Generate trials
    # ------------------------------------------------------------------
    trials = []
    for angle_step in [args.angle_step]:
        angles = list(range(0, 360, angle_step))
        for cue in [1, 2]:
            for a1 in angles:
                for a2 in angles:
                    trials.append({'cue': cue, 'color1_angle': a1, 'color2_angle': a2})

    if args.n_trials is not None and args.n_trials < len(trials):
        trial_indices = sorted(
            rng.choice(len(trials), size=args.n_trials, replace=False).tolist()
        )
    else:
        trial_indices = list(range(len(trials)))
    print(f'Using {len(trial_indices)} trials out of {len(trials)}')

    # ------------------------------------------------------------------
    # Load teacher nullspace
    # ------------------------------------------------------------------
    nullspace = _load_nullspace(teacher_run_dir)   # (14, 16)
    print(f'Nullspace shape: {nullspace.shape}')

    # ------------------------------------------------------------------
    # Build teacher trajectory list
    # ------------------------------------------------------------------
    direction_indices = args.direction_indices if args.direction_indices is not None else list(range(14))

    teacher_trajs_16d = []
    teacher_labels   = []

    if args.include_unablated_teacher:
        print('\nLoading unablated teacher ...')
        t = load_or_extract_trajectories(
            teacher_run_dir, trials, extraction_device, args.n_samples,
            ablation_direction_idx=None,
            trial_indices=trial_indices,
            cache_dir=cache_dir,
            label=args.teacher_run,
            chunk_size=args.sample_chunk_size,
            include_diffusion=args.include_diffusion,
        )
        teacher_trajs_16d.append(t)
        teacher_labels.append('unablated')

    for di in direction_indices:
        print(f'\nLoading teacher ablation dir {di} ...')
        t = load_or_extract_trajectories(
            teacher_run_dir, trials, extraction_device, args.n_samples,
            ablation_direction_idx=di,
            trial_indices=trial_indices,
            cache_dir=cache_dir,
            label=args.teacher_run,
            chunk_size=args.sample_chunk_size,
            include_diffusion=args.include_diffusion,
        )
        teacher_trajs_16d.append(t)
        teacher_labels.append(f'dir_{di:02d}')

    # ------------------------------------------------------------------
    # Build student trajectory list
    # ------------------------------------------------------------------
    student_trajs_16d = []
    student_labels   = []

    if args.include_unablated_student:
        print('\nLoading unablated teacher as student column ...')
        t = load_or_extract_trajectories(
            teacher_run_dir, trials, extraction_device, args.n_samples,
            ablation_direction_idx=None,
            trial_indices=trial_indices,
            cache_dir=cache_dir,
            label=args.teacher_run + '_as_student',
            chunk_size=args.sample_chunk_size,
            include_diffusion=args.include_diffusion,
        )
        student_trajs_16d.append(t)
        student_labels.append('unablated_teacher')

    for s_run in args.student_runs:
        print(f'\nLoading student run: {s_run} ...')
        s_run_dir = results_root / s_run
        t = load_or_extract_trajectories(
            s_run_dir, trials, extraction_device, args.n_samples,
            ablation_direction_idx=None,
            trial_indices=trial_indices,
            cache_dir=cache_dir,
            label=s_run,
            chunk_size=args.sample_chunk_size,
            include_diffusion=args.include_diffusion,
        )
        student_trajs_16d.append(t)
        # Shorten label for axis readability
        short = s_run.replace('index_cued_first_diffusion_0.3_swap_', '')
        student_labels.append(short)

    # ------------------------------------------------------------------
    # Project all trajectories to nullspace: (N, S, T, 16) -> (N, S, T, 14)
    # ------------------------------------------------------------------
    teacher_trajs = [project_to_nullspace(t, nullspace) for t in teacher_trajs_16d]
    student_trajs = [project_to_nullspace(t, nullspace) for t in student_trajs_16d]
    print(f'\nProjected to nullspace. Example shape: {teacher_trajs[0].shape}')

    # ------------------------------------------------------------------
    # Compute heatmap — scaled (c free)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Computing heatmap with free scale (allow_scaling=True) ...')
    print('=' * 60)
    heatmap_scaled = compute_heatmap(
        teacher_trajs, student_trajs,
        allow_scaling=True,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs_lap=args.n_jobs,
        seed=args.seed,
        device=device,
    )
    D_scaled = heatmap_scaled['residuals']
    np.save(output_dir / 'heatmap_procrustes_scaled.npy', D_scaled)
    save_heatmap_diagnostics(heatmap_scaled, output_dir / 'diagnostics_scaled.npz')

    # ------------------------------------------------------------------
    # Compute heatmap — fixed scale (c = 1)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Computing heatmap with fixed scale (allow_scaling=False) ...')
    print('=' * 60)
    heatmap_fixed = compute_heatmap(
        teacher_trajs, student_trajs,
        allow_scaling=False,
        n_restarts=args.n_restarts,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs_lap=args.n_jobs,
        seed=args.seed,
        device=device,
    )
    D_fixed = heatmap_fixed['residuals']
    np.save(output_dir / 'heatmap_procrustes_fixed.npy', D_fixed)
    save_heatmap_diagnostics(heatmap_fixed, output_dir / 'diagnostics_fixed.npz')

    # ------------------------------------------------------------------
    # Permutation tests
    # ------------------------------------------------------------------
    print('\nRunning permutation tests ...')
    n_t, n_s = D_scaled.shape
    if n_t == n_s:
        perm_scaled = permutation_test(D_scaled, args.n_permutations, args.seed)
        perm_fixed  = permutation_test(D_fixed,  args.n_permutations, args.seed + 1)
        np.savez(
            output_dir / 'permutation_test_scaled.npz',
            observed_T=np.array(perm_scaled['observed_T']),
            null_distribution=perm_scaled['null_distribution'],
            p_value=np.array(perm_scaled['p_value']),
            diag_mean=np.array(perm_scaled['diag_mean']),
            off_diag_mean=np.array(perm_scaled['off_diag_mean']),
        )
        np.savez(
            output_dir / 'permutation_test_fixed.npz',
            observed_T=np.array(perm_fixed['observed_T']),
            null_distribution=perm_fixed['null_distribution'],
            p_value=np.array(perm_fixed['p_value']),
            diag_mean=np.array(perm_fixed['diag_mean']),
            off_diag_mean=np.array(perm_fixed['off_diag_mean']),
        )
        print(f'Scaled heatmap: T={perm_scaled["observed_T"]:.4f}, p={perm_scaled["p_value"]:.4f}')
        print(f'Fixed  heatmap: T={perm_fixed["observed_T"]:.4f},  p={perm_fixed["p_value"]:.4f}')
        plot_permutation_test(
            perm_scaled, 'Permutation test (scaled Procrustes)',
            output_dir / 'permutation_test_scaled.png',
        )
        plot_permutation_test(
            perm_fixed, 'Permutation test (fixed-scale Procrustes)',
            output_dir / 'permutation_test_fixed.png',
        )
    else:
        print(f'Skipping permutation test: heatmap is {n_t}×{n_s} (not square)')

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    vmin = min(D_scaled.min(), D_fixed.min())
    vmax = max(D_scaled.max(), D_fixed.max())

    plot_heatmap(
        D_scaled, teacher_labels, student_labels,
        title='Procrustes residual (free scale)',
        out_path=output_dir / 'heatmap_procrustes_scaled.png',
        vmin=vmin, vmax=vmax,
    )
    plot_heatmap(
        D_fixed, teacher_labels, student_labels,
        title='Procrustes residual (fixed scale, c = 1)',
        out_path=output_dir / 'heatmap_procrustes_fixed.png',
        vmin=vmin, vmax=vmax,
    )

    # Sorted by ablation direction index (rows/cols already ordered by
    # direction_indices, so this is already natural order; we make it explicit)
    sorted_row_order = list(range(n_t))
    sorted_col_order = list(range(n_s))
    D_scaled_sorted = D_scaled[np.ix_(sorted_row_order, sorted_col_order)]
    D_fixed_sorted  = D_fixed [np.ix_(sorted_row_order, sorted_col_order)]
    sorted_row_labels = [teacher_labels[i] for i in sorted_row_order]
    sorted_col_labels = [student_labels[j] for j in sorted_col_order]

    plot_heatmap(
        D_scaled_sorted, sorted_row_labels, sorted_col_labels,
        title='Procrustes residual sorted by ablation index (free scale)',
        out_path=output_dir / 'heatmap_procrustes_scaled_sorted.png',
        vmin=vmin, vmax=vmax,
    )

    plot_comparison_heatmaps(
        D_scaled, D_fixed, teacher_labels, student_labels,
        out_path=output_dir / 'heatmap_procrustes_comparison.png',
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Heatmap shape: {D_scaled.shape}')
    print(f'Scaled  — min={D_scaled.min():.4f}  max={D_scaled.max():.4f}  mean={D_scaled.mean():.4f}')
    print(f'Fixed   — min={D_fixed.min():.4f}   max={D_fixed.max():.4f}   mean={D_fixed.mean():.4f}')
    print(f'Output in: {output_dir}')


if __name__ == '__main__':
    main()
