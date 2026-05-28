#!/usr/bin/env python3
"""End-to-end sampling -> NPZ export -> PCA -> 3D plot pipeline.

This script samples neural activity from a model checkpoint (deterministic
cue/color sweep), saves a compact NPZ cache and then calls the
`run_prospective_memory_plotting.py` wrapper to produce prep and diffusion
3D PCA figures.

It re-uses the task/model loading helpers and sampling interface used in the
notebooks so outputs match the notebook-produced NPZs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from ddpm.analysis.new_analysis.export_student_sweep_cache import (
    generate_trial_combinations,
    move_to_device,
    build_override_stimulus_features_dict,
)
from ddpm.analysis.new_analysis.prospective_memory_plotting import (
    compute_3d_limits,
    make_full_step_diffusion_dataset,
    make_full_step_trajectory_dataset,
    plot_global_3d_trajectories,
)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end sampling + PCA + plotting")
    p.add_argument("--args-path", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--angle-step", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-samples", type=int, default=1, help="Parallel sample axis (kept=1 for stable prep extraction)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--n-bins", type=int, default=12)
    p.add_argument("--variance-threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args_path = str(args.args_path)
    checkpoint_path = args.checkpoint_path
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    npz_out = outdir / "sampling_cache.npz"
    if npz_out.exists() and not args.force:
        raise FileExistsError(f"Output exists: {npz_out} (use --force to overwrite)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model/task from {args_path} on {device}")
    _, task, model, *rest = generate_model_and_task_from_args_path_multiepoch(args_path, device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    trials = generate_trial_combinations(args.angle_step)
    print(f"Generated {len(trials)} deterministic trials with angle_step={args.angle_step}")

    trajectory_chunks: List[np.ndarray] = []
    # states_seq_by_prep: dict prep_idx -> list of per-trial arrays [T_p, D]
    states_seq_by_prep: Dict[int, List[np.ndarray]] = {}
    metadata_rows: List[List[float]] = []

    neural_dim = 16

    for start in range(0, len(trials), args.batch_size):
        batch_trials = trials[start : start + args.batch_size]
        batch_size = len(batch_trials)

        # Build canonical override features (includes polar and cartesian variants)
        override = build_override_stimulus_features_dict(batch_trials)

        # Build task variables and trial_info
        task_variable_dict = task.task_variable_gen.generate_variable_dict(batch_size=batch_size, override_stimulus_features_dict=override)
        trial_info = task.generate_trial_information(batch_size=batch_size, num_samples=args.num_samples, override_task_variable_information=task_variable_dict)

        prep_inputs_device = move_to_device(trial_info.prep_network_inputs, device)
        diffusion_inputs_device = move_to_device(trial_info.diffusion_network_inputs, device)

        sample_kwargs = dict(
            prep_network_inputs=prep_inputs_device,
            diffusion_network_inputs=diffusion_inputs_device,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[batch_size, args.num_samples],
            noise_scaler=1.0,
        )

        with torch.no_grad():
            prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)

        traj_key = "embedded_sample_trajectory" if "embedded_sample_trajectory" in samples_dict else "sample_trajectory"
        if traj_key not in samples_dict:
            raise RuntimeError("No trajectory key found in samples_dict")

        # extract diffusion trajectories: ensure result is [batch, T_diff, D]
        traj_arr = samples_dict[traj_key][..., :neural_dim].detach().cpu().numpy()
        # handle possible shape (batch, 1, T, D) -> take axis=1 if present
        if traj_arr.ndim == 4 and traj_arr.shape[1] == 1:
            traj_arr = traj_arr[:, 0, :, :]
        elif traj_arr.ndim == 3:
            # assume (batch, T, D)
            pass
        else:
            raise RuntimeError(f"Unexpected trajectory array shape: {traj_arr.shape}")

        trajectory_chunks.append(traj_arr)

        # extract per-prep sequences for each trial
        for prep_idx, pd in enumerate(prep_dicts):
            if "preparatory_trajectory" not in pd:
                raise KeyError(f"preparatory_trajectory missing in prep_dicts[{prep_idx}]")
            # pd['preparatory_trajectory'] expected shape: (batch, parallel_samples, T_p, D)
            prep_tensor = pd["preparatory_trajectory"].detach().cpu().numpy()
            # ensure dict key exists
            states_seq_by_prep.setdefault(prep_idx, [])

            for i in range(batch_size):
                # take parallel sample 0
                arr = prep_tensor[i, 0, :, :neural_dim].astype(np.float32)
                states_seq_by_prep[prep_idx].append(arr)

        # metadata rows
        for t in batch_trials:
            metadata_rows.append([int(t["cue"]), float(t["color1_angle"]), float(t["color2_angle"])])

        print(f"Processed batch {start // args.batch_size + 1}: trials {start}-{start + batch_size - 1}")

    all_trajectories = np.concatenate(trajectory_chunks, axis=0)
    metadata = np.asarray(metadata_rows, dtype=np.float32)

    # Save compact NPZ cache compatible with the plotting wrapper
    np.savez_compressed(
        npz_out,
        metadata=metadata,
        states_seq_by_prep=states_seq_by_prep,
        states_seq_by_diffusion=all_trajectories,
    )

    print(f"Saved sampling cache: {npz_out}")

    prep_png = outdir / "prospective_memory_global_pca_all_prep_steps_3d.png"
    diff_png = outdir / "prospective_memory_global_pca_all_diffusion_steps_3d.png"

    prep_indices = sorted(states_seq_by_prep.keys())
    prep_points, prep_labels, prep_step_counts = make_full_step_trajectory_dataset(
        states_seq_by_prep=states_seq_by_prep,
        metadata=metadata,
        prep_indices=prep_indices,
        n_bins=args.n_bins,
    )
    diff_points, diff_labels, diff_step_count = make_full_step_diffusion_dataset(
        states_seq_by_diffusion=all_trajectories,
        metadata=metadata,
        n_bins=args.n_bins,
    )

    if prep_points.shape[0] == 0:
        raise RuntimeError("No preparatory points available for shared PCA fitting")
    if diff_points.shape[0] == 0:
        raise RuntimeError("No diffusion points available for plotting")

    n_comp = min(max(3, 3), prep_points.shape[0], prep_points.shape[1])
    if n_comp < 3:
        raise RuntimeError(
            "Need at least 3 preparatory samples/features for the shared 3D PCA plot. "
            f"Got {prep_points.shape[0]} samples and {prep_points.shape[1]} features."
        )

    pca = PCA(n_components=n_comp)
    prep_coords = pca.fit_transform(prep_points)
    diff_coords = pca.transform(diff_points)

    shared_lims = compute_3d_limits(np.vstack([prep_coords[:, :3], diff_coords[:, :3]]))

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    print("Shared PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Shared PCA cumulative variance:", cumulative)
    print(
        f"Minimal PCs for threshold {args.variance_threshold:.1%}: "
        f"{int(np.searchsorted(cumulative, args.variance_threshold) + 1)}"
    )

    print("Running preparatory plot with shared PCA basis...")
    plot_global_3d_trajectories(
        pca_coords=prep_coords,
        labels=prep_labels,
        pca=pca,
        n_bins=args.n_bins,
        out_path=prep_png,
        title="All Preparatory Timesteps in Shared Global PCA Space",
        label_schema="prep",
        lims=shared_lims,
    )

    print("Running diffusion plot with shared PCA basis...")
    plot_global_3d_trajectories(
        pca_coords=diff_coords,
        labels=diff_labels,
        pca=pca,
        n_bins=args.n_bins,
        out_path=diff_png,
        title="All Diffusion Timesteps in Shared Global PCA Space",
        label_schema="diffusion",
        lims=shared_lims,
    )

    print("End-to-end run complete. Figures saved:")
    print("- ", prep_png)
    print("- ", diff_png)


if __name__ == "__main__":
    main()
