"""
Standalone trajectory geometry analysis with common cued-color binning.

This script extends the endpoint-style prep analysis by tracking the full prep
trajectory and full diffusion trajectory in one shared PCA space.

What it does:
1. Loads a DDPM model checkpoint.
2. Generates all non-swap cue/color trial combinations.
3. Optionally applies ablation (from run ablation_data or direction sweep).
4. Extracts neural states for all requested prep indices per trial in one pass.
4. Bins states by a common feature (cued-color bin) for each cue.
5. Averages states per (cue, bin, prep_idx).
6. Fits one global PCA across all time points.
7. Visualizes full prep and diffusion trajectories for each (cue, bin).

Example:
    conda activate ddpm
    python analyze_binned_prep_trajectories.py \
      --repo-root /scratch3/shaiq_home/repos/behaviour_ddpm \
      --run-path results_link_sampler/index_cued_first_diffusion_0.3_swap_recovery_8
"""

import argparse
import json
from pathlib import Path

import colorsys
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trajectory-level prep geometry in a shared PCA space."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/scratch3/shaiq_home/repos/behaviour_ddpm"),
        help="Path to repository root.",
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        default=Path("results_link_sampler/index_cued_first_diffusion_0.3_swap_recovery_8"),
        help="Path relative to repo-root for run directory containing args.yaml and state.mdl.",
    )
    parser.add_argument(
        "--prep-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Prep indices to include in trajectory.",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=30,
        help="Color angle step in degrees for trial generation.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
        help="Number of cued-color bins.",
    )
    parser.add_argument(
        "--neural-dim",
        type=int,
        default=16,
        help="Neural dimensions to slice from postprep_state.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ddpm/analysis/new_analysis/results/<run_name>/global_pca_trajectory.",
    )
    parser.add_argument(
        "--ablation-directions",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Ablation direction indices from model.behaviour_nullspace to analyze. "
            "If omitted and --all-ablation-directions is not set, the script auto-uses "
            "run ablation_data.pt when available, else healthy only."
        ),
    )
    parser.add_argument(
        "--all-ablation-directions",
        action="store_true",
        help="Analyze all directions in model.behaviour_nullspace.",
    )
    parser.add_argument(
        "--include-healthy",
        action="store_true",
        help="Also include the healthy (no-ablation) condition alongside ablated conditions.",
    )
    parser.add_argument(
        "--separate-ablation-plots",
        action="store_true",
        help=(
            "Generate separate 3D plot files for each ablation condition "
            "(healthy and/or each direction) instead of only overlaid views."
        ),
    )
    parser.add_argument(
        "--skip-combined-plots",
        action="store_true",
        help="Skip combined/overlaid plots and only write per-condition plots when used with --separate-ablation-plots.",
    )
    return parser.parse_args()


def generate_trial_combinations(angle_step, ablation_conditions):
    angles = list(range(0, 360, angle_step))
    trials = []
    for ablation_condition in ablation_conditions:
        for cue in [1, 2]:
            for color1 in angles:
                for color2 in angles:
                    trials.append(
                        {
                            "cue": cue,
                            "color1_angle": color1,
                            "color2_angle": color2,
                            "swap": False,
                            "ablation_condition": int(ablation_condition),
                        }
                    )
    return trials


def bin_angle(angle, bin_size):
    return int(angle // bin_size) % int(360 // bin_size)


def angle_to_colour(bin_idx, n_bins):
    hue = bin_idx / n_bins
    return colorsys.hsv_to_rgb(hue, 0.9, 0.9)


def build_trial_info(trial, task):
    probe_features = torch.tensor([[trial["color1_angle"], trial["color2_angle"]]]) * (
        np.pi / 180
    )
    report_features = torch.tensor([[trial["color1_angle"], trial["color2_angle"]]]) * (
        np.pi / 180
    )

    override_stimulus_features = {
        "probe_features": probe_features,
        "report_features": report_features,
    }

    override_stimulus_cart_features = {}
    for key, value in override_stimulus_features.items():
        override_stimulus_cart_features[f"{key}_cart"] = torch.stack(
            polar2cart(1.0, value), -1
        )

    override_stimulus_features_dict = {}
    override_stimulus_features_dict.update(override_stimulus_features)
    override_stimulus_features_dict.update(override_stimulus_cart_features)
    override_stimulus_features_dict["cued_item_idx"] = torch.tensor([trial["cue"] - 1])

    task_variable_dict = task.task_variable_gen.generate_variable_dict(
        batch_size=1,
        override_stimulus_features_dict=override_stimulus_features_dict,
    )

    return task.generate_trial_information(
        batch_size=1,
        num_samples=1,
        override_task_variable_information=task_variable_dict,
    )


def extract_states_all_preps_for_trial(
    trial,
    task,
    model,
    device,
    prep_indices,
    neural_dim,
    ablation_vectors,
):
    with torch.no_grad():
        trial_info = build_trial_info(trial, task)

        prep_network_inputs_device = []
        for inp in trial_info.prep_network_inputs:
            prep_network_inputs_device.append(inp.to(device) if isinstance(inp, torch.Tensor) else inp)

        diffusion_network_inputs_device = []
        for inp in trial_info.diffusion_network_inputs:
            diffusion_network_inputs_device.append(inp.to(device) if isinstance(inp, torch.Tensor) else inp)

        sample_kwargs = {
            "prep_network_inputs": prep_network_inputs_device,
            "diffusion_network_inputs": diffusion_network_inputs_device,
            "prep_epoch_durations": trial_info.prep_epoch_durations,
            "diffusion_epoch_durations": trial_info.diffusion_epoch_durations,
            "samples_shape": [1, 1],
            "noise_scaler": 1.0,
        }

        ablation_condition = int(trial["ablation_condition"])
        if ablation_condition >= 0:
            sample_kwargs["ablation_vector"] = ablation_vectors[ablation_condition]

        prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)

        out = {}
        for p in prep_indices:
            if p >= len(prep_dicts):
                raise ValueError(f"prep_idx {p} exceeds number of prep epochs {len(prep_dicts)}")
            # Shape: [batch, samples, T_prep, neural_dim]
            prep_traj = prep_dicts[p]["preparatory_trajectory"][0, 0, :, :neural_dim].cpu().numpy()
            out[p] = prep_traj

        # Shape: [batch, samples, T_diff, neural_dim]
        diff_traj = samples_dict["embedded_sample_trajectory"][0, 0, :, :neural_dim].cpu().numpy()
        return out, diff_traj


def extract_states_for_all_trials(
    trials,
    task,
    model,
    device,
    prep_indices,
    neural_dim,
    ablation_vectors,
):
    # states_seq_by_prep[p][trial_idx] has shape [T_prep, neural_dim]
    states_seq_by_prep = {p: [] for p in prep_indices}
    states_seq_by_diffusion = []
    metadata = []

    for i, trial in enumerate(trials):
        if i % 200 == 0 and i > 0:
            print(f"  trial progress: {i}/{len(trials)}")

        trial_states, trial_diffusion_states = extract_states_all_preps_for_trial(
            trial=trial,
            task=task,
            model=model,
            device=device,
            prep_indices=prep_indices,
            neural_dim=neural_dim,
            ablation_vectors=ablation_vectors,
        )

        for p in prep_indices:
            states_seq_by_prep[p].append(trial_states[p])

        states_seq_by_diffusion.append(trial_diffusion_states)

        metadata.append(
            [
                trial["cue"],
                trial["color1_angle"],
                trial["color2_angle"],
                trial["ablation_condition"],
            ]
        )

    return states_seq_by_prep, np.array(states_seq_by_diffusion), np.array(metadata)


def bin_and_average_states(states, metadata, n_bins):
    bin_size = 360.0 / n_bins
    binned = {
        1: {b: [] for b in range(n_bins)},
        2: {b: [] for b in range(n_bins)},
    }

    for i, trial_metadata in enumerate(metadata):
        cue, c1, c2 = trial_metadata[:3]
        cue = int(cue)
        cued_angle = c1 if cue == 1 else c2
        b = bin_angle(cued_angle, bin_size)
        binned[cue][b].append(states[i])

    averaged = {
        1: np.full((n_bins, states.shape[1]), np.nan),
        2: np.full((n_bins, states.shape[1]), np.nan),
    }

    for cue in [1, 2]:
        for b in range(n_bins):
            if binned[cue][b]:
                averaged[cue][b] = np.mean(binned[cue][b], axis=0)

    return averaged


def make_full_step_trajectory_dataset(states_seq_by_prep, metadata, prep_indices, n_bins):
    """Build pooled points over all preparatory timesteps for global PCA.

    labels columns:
            [prep_idx, cue, bin, prep_order, local_step, global_step, ablation_condition]
    """
    points = []
    labels = []
    prep_step_counts = {}

    global_step_offset = 0
    for prep_order, p in enumerate(prep_indices):
        seqs = states_seq_by_prep[p]
        if len(seqs) == 0:
            prep_step_counts[int(p)] = 0
            continue

        # Assume fixed prep duration for this epoch; use first trial as reference.
        num_steps = int(seqs[0].shape[0])
        prep_step_counts[int(p)] = num_steps

        for local_step in range(num_steps):
            # Stack this local prep step across trials -> [n_trials, neural_dim]
            states_step = np.stack([seq[local_step] for seq in seqs], axis=0)
            condition_values = metadata[:, 3]
            for ablation_condition in np.unique(condition_values):
                cond_mask = condition_values == ablation_condition
                averaged = bin_and_average_states(states_step[cond_mask], metadata[cond_mask], n_bins)

                for cue in [1, 2]:
                    for b in range(n_bins):
                        vec = averaged[cue][b]
                        if np.isnan(vec).any():
                            continue
                        points.append(vec)
                        labels.append([
                            int(p),
                            int(cue),
                            int(b),
                            int(prep_order),
                            int(local_step),
                            int(global_step_offset + local_step),
                            int(ablation_condition),
                        ])

        global_step_offset += num_steps

    points = np.array(points)
    labels = np.array(labels, dtype=int)
    return points, labels, prep_step_counts


def make_full_timeline_dataset(states_seq_by_prep, states_seq_by_diffusion, metadata, prep_indices, n_bins):
    """Build pooled points over the full prep timeline and full diffusion timeline.

    labels columns:
            prep labels: [time_idx, cue, bin, local_step, global_step, phase, ablation_condition]
            diffusion labels: [time_idx, cue, bin, local_step, global_step, phase, ablation_condition]
    """
    points = []
    labels = []
    prep_time_counts = {}
    diffusion_time_count = 0

    if len(prep_indices) == 0:
        return np.array(points), np.array(labels, dtype=int), prep_time_counts, diffusion_time_count

    n_trials = len(metadata)
    full_prep_sequences = []
    for trial_idx in range(n_trials):
        trial_segments = []
        for p in prep_indices:
            trial_segments.append(states_seq_by_prep[p][trial_idx])
        full_prep_sequences.append(np.concatenate(trial_segments, axis=0))

    prep_total_steps = int(full_prep_sequences[0].shape[0])
    prep_time_counts["total"] = prep_total_steps

    for local_step in range(prep_total_steps):
        states_step = np.stack([seq[local_step] for seq in full_prep_sequences], axis=0)
        condition_values = metadata[:, 3]
        for ablation_condition in np.unique(condition_values):
            cond_mask = condition_values == ablation_condition
            averaged = bin_and_average_states(states_step[cond_mask], metadata[cond_mask], n_bins)

            for cue in [1, 2]:
                for b in range(n_bins):
                    vec = averaged[cue][b]
                    if np.isnan(vec).any():
                        continue
                    points.append(vec)
                    labels.append([
                        int(local_step),
                        int(cue),
                        int(b),
                        int(local_step),
                        int(local_step),
                        0,
                        int(ablation_condition),
                    ])

    if states_seq_by_diffusion.shape[0] > 0:
        diffusion_time_count = int(states_seq_by_diffusion.shape[1])
        for diff_step in range(diffusion_time_count):
            states_step = states_seq_by_diffusion[:, diff_step, :]
            condition_values = metadata[:, 3]
            for ablation_condition in np.unique(condition_values):
                cond_mask = condition_values == ablation_condition
                averaged = bin_and_average_states(states_step[cond_mask], metadata[cond_mask], n_bins)

                for cue in [1, 2]:
                    for b in range(n_bins):
                        vec = averaged[cue][b]
                        if np.isnan(vec).any():
                            continue
                        points.append(vec)
                        labels.append([
                            int(diff_step),
                            int(cue),
                            int(b),
                            int(diff_step),
                            int(prep_total_steps + diff_step),
                            1,
                            int(ablation_condition),
                        ])

    points = np.array(points)
    labels = np.array(labels, dtype=int)
    return points, labels, prep_time_counts, diffusion_time_count


def make_diffusion_step_dataset(states_seq_by_diffusion, metadata, n_bins):
    """Build pooled points over all diffusion timesteps for a global diffusion PCA.

    labels columns:
            [diff_step, cue, bin, global_step, ablation_condition]
    """
    points = []
    labels = []

    if states_seq_by_diffusion.shape[0] == 0:
        return np.array(points), np.array(labels, dtype=int), 0

    num_diff_steps = int(states_seq_by_diffusion.shape[1])

    for diff_step in range(num_diff_steps):
        states_step = states_seq_by_diffusion[:, diff_step, :]
        condition_values = metadata[:, 3]
        for ablation_condition in np.unique(condition_values):
            cond_mask = condition_values == ablation_condition
            averaged = bin_and_average_states(states_step[cond_mask], metadata[cond_mask], n_bins)

            for cue in [1, 2]:
                for b in range(n_bins):
                    vec = averaged[cue][b]
                    if np.isnan(vec).any():
                        continue
                    points.append(vec)
                    labels.append([
                        int(diff_step),
                        int(cue),
                        int(b),
                        int(diff_step),
                        int(ablation_condition),
                    ])

    points = np.array(points)
    labels = np.array(labels, dtype=int)
    return points, labels, num_diff_steps


def plot_all_prep_steps_3d(pca_coords, labels, pca, out_path, n_bins):
    """Plot full stepwise preparatory trajectories in one global PCA space."""
    pca_coords = pca_coords - pca_coords.mean(axis=0, keepdims=True)
    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw full trajectory for each (cue, bin), sorted by global_step.
    ablation_conditions = np.unique(labels[:, 6])
    for ablation_condition in ablation_conditions:
        for cue in [1, 2]:
            marker = "o" if cue == 1 else "^"
            for b in range(n_bins):
                mask = (
                    (labels[:, 1] == cue)
                    & (labels[:, 2] == b)
                    & (labels[:, 6] == ablation_condition)
                )
                seq = pca_coords[mask]
                seq_labels = labels[mask]
                if seq.shape[0] < 2:
                    continue

                order = np.argsort(seq_labels[:, 5])
                seq = seq[order]

                color = angle_to_colour(int(b), n_bins)
                ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], color=color, alpha=0.65, linewidth=1.8)
                ax.scatter(
                    seq[:, 0],
                    seq[:, 1],
                    seq[:, 2],
                    c=[color],
                    marker=marker,
                    s=26,
                    edgecolors="k",
                    linewidths=0.35,
                    alpha=0.9,
                )

                # Highlight start and end points for readability.
                ax.scatter(
                    seq[0, 0], seq[0, 1], seq[0, 2], c=[color], marker=marker,
                    s=72, edgecolors="k", linewidths=0.7, alpha=1.0
                )
                ax.scatter(
                    seq[-1, 0], seq[-1, 1], seq[-1, 2], c=[color], marker=marker,
                    s=100, edgecolors="k", linewidths=0.9, alpha=1.0
                )

    g = pca.explained_variance_ratio_
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xlabel(f"PC1 ({g[0]:.1%})", fontweight="bold")
    ax.set_ylabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
    ax.set_zlabel(f"PC3 ({g[2]:.1%})", fontweight="bold")
    ax.set_title("All Preparatory Timesteps in Global PCA Space", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=24, azim=-56)

    legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 1"),
        Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 2"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=6, label="Start"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=9, label="End"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_all_diffusion_steps_3d(pca_coords, labels, pca, out_path, n_bins):
    """Plot full stepwise diffusion trajectories in one diffusion-global PCA space."""
    pca_coords = pca_coords - pca_coords.mean(axis=0, keepdims=True)
    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ablation_conditions = np.unique(labels[:, 4])
    for ablation_condition in ablation_conditions:
        for cue in [1, 2]:
            marker = "o" if cue == 1 else "^"
            for b in range(n_bins):
                mask = (
                    (labels[:, 1] == cue)
                    & (labels[:, 2] == b)
                    & (labels[:, 4] == ablation_condition)
                )
                seq = pca_coords[mask]
                seq_labels = labels[mask]
                if seq.shape[0] < 2:
                    continue

                order = np.argsort(seq_labels[:, 3])
                seq = seq[order]
                color = angle_to_colour(int(b), n_bins)

                ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], color=color, alpha=0.65, linewidth=1.7)
                ax.scatter(
                    seq[:, 0],
                    seq[:, 1],
                    seq[:, 2],
                    c=[color],
                    marker=marker,
                    s=20,
                    edgecolors="k",
                    linewidths=0.3,
                    alpha=0.85,
                )

                ax.scatter(
                    seq[0, 0], seq[0, 1], seq[0, 2], c=[color], marker=marker,
                    s=68, edgecolors="k", linewidths=0.7, alpha=1.0
                )
                ax.scatter(
                    seq[-1, 0], seq[-1, 1], seq[-1, 2], c=[color], marker=marker,
                    s=92, edgecolors="k", linewidths=0.9, alpha=1.0
                )

    g = pca.explained_variance_ratio_
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xlabel(f"PC1 ({g[0]:.1%})", fontweight="bold")
    ax.set_ylabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
    ax.set_zlabel(f"PC3 ({g[2]:.1%})", fontweight="bold")
    ax.set_title("All Diffusion Timesteps in Global PCA Space", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=22, azim=-58)

    legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 1"),
        Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 2"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=6, label="Start"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=9, label="End"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_rings_by_time_3d(
    pca_coords,
    labels,
    pca,
    out_path,
    n_bins,
    time_col,
    cue_col,
    bin_col,
    condition_col,
    title,
):
    """At each fixed timestep, connect points across bins; darken lines over time."""
    pca_coords = pca_coords - pca_coords.mean(axis=0, keepdims=True)
    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    times = np.unique(labels[:, time_col])
    if len(times) == 1:
        t_min, t_max = float(times[0]), float(times[0] + 1.0)
    else:
        t_min, t_max = float(times.min()), float(times.max())

    # Build a lookup so we can retrieve coordinates by (time, cue, bin, condition)
    lookup = {}
    for i in range(labels.shape[0]):
        t = int(labels[i, time_col])
        cue = int(labels[i, cue_col])
        b = int(labels[i, bin_col])
        condition = int(labels[i, condition_col])
        lookup[(t, cue, b, condition)] = pca_coords[i]

    condition_values = np.unique(labels[:, condition_col])

    for t in times:
        # Darker as time increases.
        norm = (float(t) - t_min) / max(t_max - t_min, 1e-12)
        gray = 0.85 - 0.75 * norm
        line_color = (gray, gray, gray)

        for condition in condition_values:
            for cue in [1, 2]:
                pts = []
                for b in range(n_bins):
                    key = (int(t), cue, int(b), int(condition))
                    if key in lookup:
                        pts.append(lookup[key])

                if len(pts) < 3:
                    continue

                pts = np.array(pts)
                # Close the ring.
                closed = np.vstack([pts, pts[0:1]])

                ax.plot(
                    closed[:, 0],
                    closed[:, 1],
                    closed[:, 2],
                    color=line_color,
                    linewidth=1.3 if cue == 1 else 1.0,
                    linestyle="-" if cue == 1 else "--",
                    alpha=0.95,
                )

    g = pca.explained_variance_ratio_
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xlabel(f"PC1 ({g[0]:.1%})", fontweight="bold")
    ax.set_ylabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
    ax.set_zlabel(f"PC3 ({g[2]:.1%})", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=24, azim=-56)

    legend = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Colour 1 ring"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Colour 2 ring"),
        Line2D([0], [0], color=(0.85, 0.85, 0.85), linestyle="-", linewidth=2, label="Early time"),
        Line2D([0], [0], color=(0.10, 0.10, 0.10), linestyle="-", linewidth=2, label="Late time"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_geometry(pca_coords, labels, prep_indices, pca, out_path, n_bins):
    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]

    fig = plt.figure(figsize=(19, 6.5))
    ax3d = fig.add_subplot(131, projection="3d")
    ax12 = fig.add_subplot(132)
    ax23 = fig.add_subplot(133)

    # Draw trajectory lines for each (cue, bin) through prep_order
    for cue in [1, 2]:
        marker = "o" if cue == 1 else "^"
        for b in range(n_bins):
            mask = (labels[:, 1] == cue) & (labels[:, 2] == b)
            seq = pca_coords[mask]
            seq_labels = labels[mask]
            if seq.shape[0] < 2:
                continue

            # Sort by prep order so lines connect in temporal order
            order = np.argsort(seq_labels[:, 3])
            seq = seq[order]

            color = angle_to_colour(int(b), n_bins)

            ax3d.plot(seq[:, 0], seq[:, 1], seq[:, 2], color=color, alpha=0.55, linewidth=1.6)
            ax12.plot(seq[:, 0], seq[:, 1], color=color, alpha=0.55, linewidth=1.6)
            ax23.plot(seq[:, 1], seq[:, 2], color=color, alpha=0.55, linewidth=1.6)

            ax3d.scatter(seq[:, 0], seq[:, 1], seq[:, 2], c=[color], marker=marker, s=42, edgecolors="k", linewidths=0.5)
            ax12.scatter(seq[:, 0], seq[:, 1], c=[color], marker=marker, s=42, edgecolors="k", linewidths=0.5)
            ax23.scatter(seq[:, 1], seq[:, 2], c=[color], marker=marker, s=42, edgecolors="k", linewidths=0.5)

    g = pca.explained_variance_ratio_

    ax3d.set_xlim(lims[0])
    ax3d.set_ylim(lims[1])
    ax3d.set_zlim(lims[2])
    ax3d.set_xlabel(f"PC1 ({g[0]:.1%})")
    ax3d.set_ylabel(f"PC2 ({g[1]:.1%})")
    ax3d.set_zlabel(f"PC3 ({g[2]:.1%})")
    ax3d.set_title("3D Trajectories (lines connect prep_idx)", fontweight="bold")
    ax3d.grid(True, alpha=0.25)

    cue_legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 1"),
        Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 2"),
    ]
    ax3d.legend(handles=cue_legend, fontsize=8, loc="upper right")

    ax12.set_xlim(lims[0])
    ax12.set_ylim(lims[1])
    ax12.set_xlabel(f"PC1 ({g[0]:.1%})", fontweight="bold")
    ax12.set_ylabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
    ax12.set_title("PC1 vs PC2 Trajectories", fontweight="bold")
    ax12.grid(True, alpha=0.25)
    ax12.set_aspect("equal", adjustable="box")

    ax23.set_xlim(lims[1])
    ax23.set_ylim(lims[2])
    ax23.set_xlabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
    ax23.set_ylabel(f"PC3 ({g[2]:.1%})", fontweight="bold")
    ax23.set_title("PC2 vs PC3 Trajectories", fontweight="bold")
    ax23.grid(True, alpha=0.25)
    ax23.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_adjacent_segment_transitions(pca_coords, labels, prep_indices, pca, out_path, n_bins):
    """Plot one panel per adjacent prep transition (e.g. 0->1, 1->2, 2->3)."""
    if len(prep_indices) < 2:
        raise ValueError("Need at least two prep indices for transition plotting.")

    segment_pairs = list(zip(prep_indices[:-1], prep_indices[1:]))

    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    xlim = (mins[0] - pad[0], maxs[0] + pad[0])
    ylim = (mins[1] - pad[1], maxs[1] + pad[1])
    zlim = (mins[2] - pad[2], maxs[2] + pad[2])

    fig = plt.figure(figsize=(7.2 * len(segment_pairs), 6.8))
    axes = [
        fig.add_subplot(1, len(segment_pairs), i + 1, projection="3d")
        for i in range(len(segment_pairs))
    ]

    g = pca.explained_variance_ratio_

    # Build quick lookup from (prep_idx, cue, bin) -> PC coords
    coord_lookup = {}
    for i in range(labels.shape[0]):
        prep_idx = int(labels[i, 0])
        cue = int(labels[i, 1])
        bin_idx = int(labels[i, 2])
        coord_lookup[(prep_idx, cue, bin_idx)] = pca_coords[i]

    for ax, (p0, p1) in zip(axes, segment_pairs):
        for cue in [1, 2]:
            marker = "o" if cue == 1 else "^"
            for b in range(n_bins):
                k0 = (int(p0), cue, int(b))
                k1 = (int(p1), cue, int(b))
                if k0 not in coord_lookup or k1 not in coord_lookup:
                    continue

                c0 = coord_lookup[k0]
                c1 = coord_lookup[k1]
                color = angle_to_colour(int(b), n_bins)

                ax.plot(
                    [c0[0], c1[0]],
                    [c0[1], c1[1]],
                    [c0[2], c1[2]],
                    color=color,
                    alpha=0.7,
                    linewidth=2.0,
                )
                ax.scatter(
                    c0[0],
                    c0[1],
                    c0[2],
                    c=[color],
                    marker=marker,
                    s=58,
                    edgecolors="k",
                    linewidths=0.6,
                    alpha=0.95,
                )
                ax.scatter(
                    c1[0],
                    c1[1],
                    c1[2],
                    c=[color],
                    marker=marker,
                    s=95,
                    edgecolors="k",
                    linewidths=0.8,
                    alpha=0.95,
                )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel(f"PC1 ({g[0]:.1%})", fontweight="bold")
        ax.set_ylabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
        ax.set_zlabel(f"PC3 ({g[2]:.1%})", fontweight="bold")
        ax.set_title(f"Transition {p0}->{p1}", fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.view_init(elev=23, azim=-54)

    cue_legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 1"),
        Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 2"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=6, label="Start"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=9, label="End"),
    ]
    axes[-1].legend(handles=cue_legend, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_time_trajectory_3d(
    pca_coords,
    labels,
    pca,
    out_path,
    n_bins,
    time_col,
    cue_col,
    bin_col,
    condition_col,
    title,
):
    """Plot a 3D trajectory where time order is encoded by connected points."""
    pca_coords = pca_coords - pca_coords.mean(axis=0, keepdims=True)
    mins = pca_coords.min(axis=0)
    maxs = pca_coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    times = np.unique(labels[:, time_col])
    t_min = float(times.min())
    t_max = float(times.max()) if len(times) > 1 else float(times.min() + 1.0)

    for condition in np.unique(labels[:, condition_col]):
        for cue in [1, 2]:
            marker = "o" if cue == 1 else "^"
            for b in range(n_bins):
                mask = (
                    (labels[:, cue_col] == cue)
                    & (labels[:, bin_col] == b)
                    & (labels[:, condition_col] == condition)
                )
                seq = pca_coords[mask]
                seq_labels = labels[mask]
                if seq.shape[0] < 2:
                    continue

                order = np.argsort(seq_labels[:, time_col])
                seq = seq[order]
                seq_times = seq_labels[order, time_col]
                color = angle_to_colour(int(b), n_bins)

                ax.plot(seq[:, 0], seq[:, 1], seq[:, 2], color=color, alpha=0.55, linewidth=1.6)
                time_norm = (seq_times - t_min) / max(t_max - t_min, 1e-12)
                point_colors = [(0.2 + 0.75 * float(t), 0.2 + 0.55 * float(t), 0.2 + 0.35 * float(t)) for t in time_norm]
                ax.scatter(
                    seq[:, 0],
                    seq[:, 1],
                    seq[:, 2],
                    c=point_colors,
                    marker=marker,
                    s=26,
                    edgecolors="k",
                    linewidths=0.35,
                    alpha=0.95,
                )
                ax.scatter(
                    seq[0, 0], seq[0, 1], seq[0, 2],
                    c=[color], marker=marker, s=72, edgecolors="k", linewidths=0.7, alpha=1.0
                )
                ax.scatter(
                    seq[-1, 0], seq[-1, 1], seq[-1, 2],
                    c=[color], marker=marker, s=100, edgecolors="k", linewidths=0.9, alpha=1.0
                )

    g = pca.explained_variance_ratio_
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xlabel(f"PC1 ({g[0]:.1%})", fontweight="bold")
    ax.set_ylabel(f"PC2 ({g[1]:.1%})", fontweight="bold")
    ax.set_zlabel(f"PC3 ({g[2]:.1%})", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.view_init(elev=24, azim=-56)

    legend = [
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 1"),
        Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 2"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=6, label="Start"),
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=9, label="End"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def compute_prep_local_variance_fractions(pca_coords, labels, prep_indices):
    out = {}
    for p in prep_indices:
        mask = labels[:, 0] == p
        x = pca_coords[mask]
        if x.shape[0] > 1:
            var = x.var(axis=0, ddof=1)
            frac = var / max(var.sum(), 1e-12)
            out[int(p)] = {
                "PC1": float(frac[0]),
                "PC2": float(frac[1]),
                "PC3": float(frac[2]),
            }
        else:
            out[int(p)] = {"PC1": 0.0, "PC2": 0.0, "PC3": 0.0}
    return out


def resolve_run_dir(repo_root, run_path):
    run_path = Path(run_path)

    candidate_dirs = []
    if run_path.is_absolute():
        candidate_dirs.append(run_path)
    else:
        candidate_dirs.append((repo_root / run_path).resolve())
        for prefix in ["results_link_sampler", "results_link_sampler_ext", "results_link_drl"]:
            candidate_dirs.append((repo_root / prefix / run_path).resolve())

    seen = set()
    unique_candidate_dirs = []
    for path in candidate_dirs:
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            unique_candidate_dirs.append(path)

    for candidate in unique_candidate_dirs:
        args_path = candidate / "args.yaml"
        checkpoint_path = candidate / "state.mdl"
        if args_path.exists() and checkpoint_path.exists():
            return candidate

    checked = "\n  - " + "\n  - ".join(str(p) for p in unique_candidate_dirs)
    raise FileNotFoundError(
        "Could not resolve --run-path to a model run directory containing both "
        f"args.yaml and state.mdl. Checked:{checked}"
    )


def load_run_ablation_vector(run_dir, device):
    ablation_data_path = run_dir / "ablation_data.pt"
    if not ablation_data_path.exists():
        return None, None

    ablation_data = torch.load(ablation_data_path, map_location=device, weights_only=True)
    if "ablation_vector" not in ablation_data:
        return None, None

    vec = ablation_data["ablation_vector"].to(device)
    vec = vec / torch.norm(vec)
    neuron_idx = ablation_data.get("neuron_idx", None)
    return vec, neuron_idx


def build_ablation_vectors(model, direction_indices, device):
    ablation_vectors = {}
    for direction_idx in direction_indices:
        vec = model.behaviour_nullspace[direction_idx].clone()
        vec = vec / torch.norm(vec)
        ablation_vectors[int(direction_idx)] = vec.to(device)
    return ablation_vectors


def get_ablation_setup(args, model, run_dir, device):
    if args.ablation_directions is not None and args.all_ablation_directions:
        raise ValueError("Use either --ablation-directions or --all-ablation-directions, not both.")

    ablation_vectors = {}
    run_ablation_vector, run_ablation_idx = load_run_ablation_vector(run_dir, device)

    if args.all_ablation_directions or args.ablation_directions is not None:
        if not hasattr(model, "behaviour_nullspace"):
            raise ValueError(
                "Model has no behaviour_nullspace, so ablation direction sweep is unavailable."
            )

        max_dirs = model.behaviour_nullspace.shape[0]
        if args.all_ablation_directions:
            direction_indices = list(range(max_dirs))
        else:
            direction_indices = sorted(set(int(d) for d in args.ablation_directions))
            bad = [d for d in direction_indices if d < 0 or d >= max_dirs]
            if bad:
                raise ValueError(
                    f"Invalid ablation direction(s) {bad}; valid range is [0, {max_dirs - 1}]"
                )

        ablation_vectors.update(build_ablation_vectors(model, direction_indices, device))
        ablation_conditions = list(direction_indices)
        source = "direction_sweep"
    elif run_ablation_vector is not None:
        condition_idx = int(run_ablation_idx) if run_ablation_idx is not None else 0
        if condition_idx < 0:
            condition_idx = 0
        ablation_vectors[condition_idx] = run_ablation_vector
        ablation_conditions = [condition_idx]
        source = "run_ablation_data"
    else:
        ablation_conditions = []
        source = "healthy_only"

    if args.include_healthy or not ablation_conditions:
        all_conditions = [-1] + ablation_conditions
    else:
        all_conditions = ablation_conditions

    info = {
        "source": source,
        "run_ablation_neuron_idx": None if run_ablation_idx is None else int(run_ablation_idx),
        "run_has_ablation_data": run_ablation_vector is not None,
    }
    return all_conditions, ablation_vectors, info


def ablation_condition_tag(condition):
    condition = int(condition)
    if condition < 0:
        return "healthy"
    return f"dir_{condition}"


def main():
    args = parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = resolve_run_dir(repo_root, args.run_path)
    args_path = run_dir / "args.yaml"
    checkpoint_path = run_dir / "state.mdl"

    if args.output_dir is None:
        output_dir = (
            repo_root
            / "ddpm"
            / "analysis"
            / "new_analysis"
            / "results"
            / run_dir.name
            / "global_pca_trajectory"
        )
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Repo root: {repo_root}")
    print(f"Run dir: {run_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(args_path), device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    ablation_conditions, ablation_vectors, ablation_info = get_ablation_setup(
        args=args,
        model=model,
        run_dir=run_dir,
        device=device,
    )
    print(
        f"Ablation setup: source={ablation_info['source']}, "
        f"conditions={ablation_conditions}"
    )

    prep_indices = list(args.prep_indices)
    print(f"Prep indices: {prep_indices}")

    trials = generate_trial_combinations(args.angle_step, ablation_conditions)
    print(
        f"Generated {len(trials)} trials "
        f"({len(ablation_conditions)} ablation conditions: {ablation_conditions})"
    )

    print("Extracting all preparatory timesteps across trials...")
    states_seq_by_prep, states_seq_by_diffusion, metadata = extract_states_for_all_trials(
        trials=trials,
        task=task,
        model=model,
        device=device,
        prep_indices=prep_indices,
        neural_dim=args.neural_dim,
        ablation_vectors=ablation_vectors,
    )

    print("Binning and averaging each preparatory timestep and diffusion timestep by cued-color bin...")
    timeline_points, timeline_labels, prep_time_counts, num_diff_steps = make_full_timeline_dataset(
        states_seq_by_prep=states_seq_by_prep,
        states_seq_by_diffusion=states_seq_by_diffusion,
        metadata=metadata,
        prep_indices=prep_indices,
        n_bins=args.n_bins,
    )

    pca_global = PCA(n_components=3)
    pca_coords = pca_global.fit_transform(timeline_points)

    print("Global PCA explained variance ratio:", pca_global.explained_variance_ratio_)
    print("Global PCA cumulative variance:", pca_global.explained_variance_ratio_.cumsum())

    prep_mask = timeline_labels[:, 5] == 0
    diff_mask = timeline_labels[:, 5] == 1
    prep_coords = pca_coords[prep_mask]
    prep_labels = timeline_labels[prep_mask]
    diff_coords = pca_coords[diff_mask]
    diff_labels = timeline_labels[diff_mask]

    prep_fig_path = output_dir / "retrospective_memory_global_pca_all_prep_timesteps_3d.png"
    diff_fig_path = output_dir / "retrospective_memory_global_pca_all_diffusion_timesteps_3d.png"
    prep_ring_fig_path = output_dir / "retrospective_memory_global_pca_all_prep_timesteps_3d_rings.png"
    diff_ring_fig_path = output_dir / "retrospective_memory_global_pca_all_diffusion_timesteps_3d_rings.png"

    if not args.skip_combined_plots:
        plot_time_trajectory_3d(
            pca_coords=prep_coords,
            labels=prep_labels,
            pca=pca_global,
            out_path=prep_fig_path,
            n_bins=args.n_bins,
            time_col=0,
            cue_col=1,
            bin_col=2,
            condition_col=6,
            title="Retrospective colours-first, cue-later: all preparatory timesteps in global PCA space",
        )

        plot_time_trajectory_3d(
            pca_coords=diff_coords,
            labels=diff_labels,
            pca=pca_global,
            out_path=diff_fig_path,
            n_bins=args.n_bins,
            time_col=0,
            cue_col=1,
            bin_col=2,
            condition_col=6,
            title="Retrospective colours-first, cue-later: all diffusion timesteps in global PCA space",
        )

        plot_spatial_rings_by_time_3d(
            pca_coords=prep_coords,
            labels=prep_labels,
            pca=pca_global,
            out_path=prep_ring_fig_path,
            n_bins=args.n_bins,
            time_col=0,
            cue_col=1,
            bin_col=2,
            condition_col=6,
            title="Retrospective colours-first ring view: all preparatory timesteps in global PCA space",
        )

        plot_spatial_rings_by_time_3d(
            pca_coords=diff_coords,
            labels=diff_labels,
            pca=pca_global,
            out_path=diff_ring_fig_path,
            n_bins=args.n_bins,
            time_col=0,
            cue_col=1,
            bin_col=2,
            condition_col=6,
            title="Retrospective colours-first ring view: all diffusion timesteps in global PCA space",
        )

    separate_figures = {}
    if args.separate_ablation_plots:
        for condition in ablation_conditions:
            cond_tag = ablation_condition_tag(condition)
            cond_mask = timeline_labels[:, 6] == int(condition)
            cond_coords = pca_coords[cond_mask]
            cond_labels = timeline_labels[cond_mask]

            if cond_coords.shape[0] == 0:
                continue

            cond_prep_mask = cond_labels[:, 5] == 0
            cond_diff_mask = cond_labels[:, 5] == 1
            cond_prep_coords = cond_coords[cond_prep_mask]
            cond_prep_labels = cond_labels[cond_prep_mask]
            cond_diff_coords = cond_coords[cond_diff_mask]
            cond_diff_labels = cond_labels[cond_diff_mask]

            if cond_prep_coords.shape[0] == 0 or cond_diff_coords.shape[0] == 0:
                continue

            cond_prep_path = output_dir / f"retrospective_memory_global_pca_all_prep_timesteps_3d_{cond_tag}.png"
            cond_diff_path = output_dir / f"retrospective_memory_global_pca_all_diffusion_timesteps_3d_{cond_tag}.png"
            cond_prep_ring_path = output_dir / f"retrospective_memory_global_pca_all_prep_timesteps_3d_rings_{cond_tag}.png"
            cond_diff_ring_path = output_dir / f"retrospective_memory_global_pca_all_diffusion_timesteps_3d_rings_{cond_tag}.png"

            plot_time_trajectory_3d(
                pca_coords=cond_prep_coords,
                labels=cond_prep_labels,
                pca=pca_global,
                out_path=cond_prep_path,
                n_bins=args.n_bins,
                time_col=0,
                cue_col=1,
                bin_col=2,
                condition_col=6,
                title=f"Retrospective colours-first, cue-later: all preparatory timesteps in global PCA space ({cond_tag})",
            )
            plot_time_trajectory_3d(
                pca_coords=cond_diff_coords,
                labels=cond_diff_labels,
                pca=pca_global,
                out_path=cond_diff_path,
                n_bins=args.n_bins,
                time_col=0,
                cue_col=1,
                bin_col=2,
                condition_col=6,
                title=f"Retrospective colours-first, cue-later: all diffusion timesteps in global PCA space ({cond_tag})",
            )
            plot_spatial_rings_by_time_3d(
                pca_coords=cond_prep_coords,
                labels=cond_prep_labels,
                pca=pca_global,
                out_path=cond_prep_ring_path,
                n_bins=args.n_bins,
                time_col=0,
                cue_col=1,
                bin_col=2,
                condition_col=6,
                title=f"Retrospective colours-first ring view: all preparatory timesteps in global PCA space ({cond_tag})",
            )
            plot_spatial_rings_by_time_3d(
                pca_coords=cond_diff_coords,
                labels=cond_diff_labels,
                pca=pca_global,
                out_path=cond_diff_ring_path,
                n_bins=args.n_bins,
                time_col=0,
                cue_col=1,
                bin_col=2,
                condition_col=6,
                title=f"Retrospective colours-first ring view: all diffusion timesteps in global PCA space ({cond_tag})",
            )

            separate_figures[str(cond_tag)] = {
                "prep_path": str(cond_prep_path),
                "diffusion_path": str(cond_diff_path),
                "prep_ring_path": str(cond_prep_ring_path),
                "diffusion_ring_path": str(cond_diff_ring_path),
            }

    local_fractions = compute_prep_local_variance_fractions(
        pca_coords=pca_coords,
        labels=timeline_labels,
        prep_indices=prep_indices,
    )

    np.savez(
        output_dir / "global_pca_trajectory_data.npz",
        points=timeline_points,
        labels=timeline_labels,
        pca_coords=pca_coords,
        pca_components=pca_global.components_,
        pca_mean=pca_global.mean_,
        explained_variance=pca_global.explained_variance_ratio_,
    )


    summary = {
        "run_dir": str(run_dir),
        "prep_indices": [int(p) for p in prep_indices],
        "n_trials": int(len(trials)),
        "ablation_conditions": [int(c) for c in ablation_conditions],
        "healthy_condition_code": -1,
        "ablation_info": ablation_info,
        "separate_ablation_plots": bool(args.separate_ablation_plots),
        "skip_combined_plots": bool(args.skip_combined_plots),
        "separate_figure_paths": separate_figures,
        "n_bins": int(args.n_bins),
        "angle_step": int(args.angle_step),
        "neural_dim": int(args.neural_dim),
        "global_explained_variance_ratio": pca_global.explained_variance_ratio_.tolist(),
        "global_explained_variance_cumulative": pca_global.explained_variance_ratio_.cumsum().tolist(),
        "per_prep_local_variance_fraction_in_global_basis": local_fractions,
        "prep_step_counts": prep_time_counts,
        "prep_time_step_count": int(prep_time_counts.get("total", 0)),
        "diffusion_step_count": int(num_diff_steps),
        "figure_path": str(prep_fig_path),
        "diffusion_figure_path": str(diff_fig_path),
        "prep_ring_figure_path": str(prep_ring_fig_path),
        "diffusion_ring_figure_path": str(diff_ring_fig_path),
    }

    with open(output_dir / "global_pca_trajectory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs:")
    if not args.skip_combined_plots:
        print(f"  {prep_fig_path}")
        print(f"  {diff_fig_path}")
        print(f"  {prep_ring_fig_path}")
        print(f"  {diff_ring_fig_path}")
    for cond_tag, cond_paths in separate_figures.items():
        print(f"  [{cond_tag}] {cond_paths['prep_path']}")
        print(f"  [{cond_tag}] {cond_paths['diffusion_path']}")
        print(f"  [{cond_tag}] {cond_paths['prep_ring_path']}")
        print(f"  [{cond_tag}] {cond_paths['diffusion_ring_path']}")
    print(f"  {output_dir / 'global_pca_trajectory_data.npz'}")
    print(f"  {output_dir / 'global_pca_trajectory_summary.json'}")


if __name__ == "__main__":
    main()
