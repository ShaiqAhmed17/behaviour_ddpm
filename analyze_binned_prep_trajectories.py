"""
Standalone trajectory geometry analysis with common cued-color binning.

This script extends the endpoint-style prep analysis by tracking the full prep
trajectory across multiple prep indices in one shared PCA space.

What it does:
1. Loads a DDPM model checkpoint.
2. Generates all non-swap cue/color trial combinations.
3. Extracts neural states for all requested prep indices per trial in one pass.
4. Bins states by a common feature (cued-color bin) for each cue.
5. Averages states per (cue, bin, prep_idx).
6. Fits one global PCA across all prep_idx points.
7. Visualizes trajectories that connect prep_idx points for each (cue, bin).

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
    return parser.parse_args()


def generate_trial_combinations(angle_step):
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for color1 in angles:
            for color2 in angles:
                trials.append(
                    {
                        "cue": cue,
                        "color1_angle": color1,
                        "color2_angle": color2,
                        "swap": False,
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


def extract_states_all_preps_for_trial(trial, task, model, device, prep_indices, neural_dim):
    with torch.no_grad():
        trial_info = build_trial_info(trial, task)

        prep_network_inputs_device = []
        for inp in trial_info.prep_network_inputs:
            prep_network_inputs_device.append(inp.to(device) if isinstance(inp, torch.Tensor) else inp)

        diffusion_network_inputs_device = []
        for inp in trial_info.diffusion_network_inputs:
            diffusion_network_inputs_device.append(inp.to(device) if isinstance(inp, torch.Tensor) else inp)

        prep_dicts, _ = model.generate_samples(
            prep_network_inputs=prep_network_inputs_device,
            diffusion_network_inputs=diffusion_network_inputs_device,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[1, 1],
            noise_scaler=1.0,
        )

        out = {}
        for p in prep_indices:
            if p >= len(prep_dicts):
                raise ValueError(f"prep_idx {p} exceeds number of prep epochs {len(prep_dicts)}")
            out[p] = prep_dicts[p]["postprep_state"][0, 0, :neural_dim].cpu().numpy()
        return out


def extract_states_for_all_trials(trials, task, model, device, prep_indices, neural_dim):
    states_by_prep = {p: [] for p in prep_indices}
    metadata = []

    for i, trial in enumerate(trials):
        if i % 200 == 0 and i > 0:
            print(f"  trial progress: {i}/{len(trials)}")

        trial_states = extract_states_all_preps_for_trial(
            trial=trial,
            task=task,
            model=model,
            device=device,
            prep_indices=prep_indices,
            neural_dim=neural_dim,
        )

        for p in prep_indices:
            states_by_prep[p].append(trial_states[p])

        metadata.append([trial["cue"], trial["color1_angle"], trial["color2_angle"]])

    for p in prep_indices:
        states_by_prep[p] = np.array(states_by_prep[p])

    return states_by_prep, np.array(metadata)


def bin_and_average_states(states, metadata, n_bins):
    bin_size = 360.0 / n_bins
    binned = {
        1: {b: [] for b in range(n_bins)},
        2: {b: [] for b in range(n_bins)},
    }

    for i, (cue, c1, c2) in enumerate(metadata):
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


def make_trajectory_dataset(states_by_prep, metadata, prep_indices, n_bins):
    averaged_by_prep = {}
    for p in prep_indices:
        averaged_by_prep[p] = bin_and_average_states(states_by_prep[p], metadata, n_bins)

    points = []
    labels = []  # [prep_idx, cue, bin, prep_order]

    for prep_order, p in enumerate(prep_indices):
        for cue in [1, 2]:
            for b in range(n_bins):
                vec = averaged_by_prep[p][cue][b]
                if np.isnan(vec).any():
                    continue
                points.append(vec)
                labels.append([p, cue, b, prep_order])

    points = np.array(points)
    labels = np.array(labels, dtype=int)
    return averaged_by_prep, points, labels


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


def main():
    args = parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = (repo_root / args.run_path).resolve()
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

    prep_indices = list(args.prep_indices)
    print(f"Prep indices: {prep_indices}")

    trials = generate_trial_combinations(args.angle_step)
    print(f"Generated {len(trials)} trials")

    print("Extracting full prep trajectories across trials...")
    states_by_prep, metadata = extract_states_for_all_trials(
        trials=trials,
        task=task,
        model=model,
        device=device,
        prep_indices=prep_indices,
        neural_dim=args.neural_dim,
    )

    print("Binning and averaging states by cued-color bin...")
    _, points, labels = make_trajectory_dataset(
        states_by_prep=states_by_prep,
        metadata=metadata,
        prep_indices=prep_indices,
        n_bins=args.n_bins,
    )

    pca_global = PCA(n_components=3)
    pca_coords = pca_global.fit_transform(points)

    print("Global PCA explained variance ratio:", pca_global.explained_variance_ratio_)
    print("Global PCA cumulative variance:", pca_global.explained_variance_ratio_.cumsum())

    fig_path = output_dir / "prospective_memory_global_pca_adjacent_segments.png"
    plot_adjacent_segment_transitions(
        pca_coords=pca_coords,
        labels=labels,
        prep_indices=prep_indices,
        pca=pca_global,
        out_path=fig_path,
        n_bins=args.n_bins,
    )

    local_fractions = compute_prep_local_variance_fractions(
        pca_coords=pca_coords,
        labels=labels,
        prep_indices=prep_indices,
    )

    np.savez(
        output_dir / "global_pca_trajectory_data.npz",
        points=points,
        labels=labels,
        pca_coords=pca_coords,
        pca_components=pca_global.components_,
        pca_mean=pca_global.mean_,
        explained_variance=pca_global.explained_variance_ratio_,
    )

    summary = {
        "run_dir": str(run_dir),
        "prep_indices": [int(p) for p in prep_indices],
        "n_trials": int(len(trials)),
        "n_bins": int(args.n_bins),
        "angle_step": int(args.angle_step),
        "neural_dim": int(args.neural_dim),
        "global_explained_variance_ratio": pca_global.explained_variance_ratio_.tolist(),
        "global_explained_variance_cumulative": pca_global.explained_variance_ratio_.cumsum().tolist(),
        "per_prep_local_variance_fraction_in_global_basis": local_fractions,
        "segment_pairs": [[int(prep_indices[i]), int(prep_indices[i + 1])] for i in range(len(prep_indices) - 1)],
        "figure_path": str(fig_path),
    }

    with open(output_dir / "global_pca_trajectory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs:")
    print(f"  {fig_path}")
    print(f"  {output_dir / 'global_pca_trajectory_data.npz'}")
    print(f"  {output_dir / 'global_pca_trajectory_summary.json'}")


if __name__ == "__main__":
    main()
