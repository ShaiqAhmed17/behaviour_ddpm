"""Single-cue prospective-memory plane fitting analysis.

This script adapts the notebook-style target/distractor plane visualization to a
single cue condition. It generates 12 x 12 = 144 trials for one cue, bins states
by target and distractor angles, fits a shared PCA, and visualizes the target
and distractor planes in 3D and 2D projections.

Example:
    conda activate ddpm
    python analyze_single_cue_target_distractor_planes.py \
      --repo-root /scratch3/shaiq_home/repos/behaviour_ddpm \
      --run-path results_link_sampler/index_cued_first_diffusion_0.3_swap_7 \
      --cue 1
"""

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-cue target/distractor plane fitting for prospective memory."
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
        default=Path("results_link_sampler/index_cued_first_diffusion_0.3_swap_7"),
        help="Path relative to repo-root for run directory containing args.yaml and state.mdl.",
    )
    parser.add_argument(
        "--cue",
        type=int,
        default=1,
        choices=[1, 2],
        help="Single cue to analyze.",
    )
    parser.add_argument(
        "--prep-idx",
        type=int,
        default=2,
        help="Preparatory epoch index used for the plane fit.",
    )
    parser.add_argument(
        "--all-prep-indices",
        action="store_true",
        help="Generate plots for all prep indices (0, 1, 2, 3) instead of a single prep_idx.",
    )
    parser.add_argument(
        "--all-timesteps",
        action="store_true",
        help="Generate plots for every model timestep (all prep and diffusion timesteps).",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=30,
        help="Angle step in degrees for target and distractor trial generation.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
        help="Number of angle bins.",
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
        help=(
            "Output directory. Defaults to ddpm/analysis/new_analysis/results/"
            "<run_name>/single_cue_target_distractor_planes."
        ),
    )
    return parser.parse_args()


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


def generate_trial_combinations(cue, angle_step):
    angles = list(range(0, 360, angle_step))
    trials = []
    for target_angle in angles:
        for distractor_angle in angles:
            if cue == 1:
                color1_angle = target_angle
                color2_angle = distractor_angle
            else:
                color1_angle = distractor_angle
                color2_angle = target_angle

            trials.append(
                {
                    "cue": int(cue),
                    "color1_angle": int(color1_angle),
                    "color2_angle": int(color2_angle),
                    "target_angle": int(target_angle),
                    "distractor_angle": int(distractor_angle),
                    "swap": False,
                }
            )
    return trials


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


def extract_state_for_trial(trial, task, model, device, prep_idx, neural_dim):
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

        prep_dicts, _ = model.generate_samples(**sample_kwargs)
        if prep_idx >= len(prep_dicts):
            raise ValueError(
                f"prep_idx {prep_idx} exceeds number of prep epochs {len(prep_dicts)}"
            )

        postprep = prep_dicts[prep_idx]["postprep_state"]
        if postprep.ndim == 3:
            state = postprep.mean(dim=1)[0]
        else:
            state = postprep[0]

        return state[:neural_dim].detach().cpu().numpy()



def extract_all_states_for_trial(trial, task, model, device, neural_dim):
    """Return ordered (label, state) pairs for all prep and diffusion timesteps for a trial."""
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

        prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)

        labels = []
        states = []

        # Prep epoch post states
        for p_idx, p_dict in enumerate(prep_dicts):
            postprep = p_dict["postprep_state"]
            if postprep.ndim == 3:
                st = postprep.mean(dim=1)[0]
            else:
                st = postprep[0]
            labels.append(f"prep_{p_idx}")
            states.append(st[:neural_dim].detach().cpu().numpy())

        # Diffusion trajectory states (if available)
        traj = None
        for key in ["embedded_sample_trajectory", "sample_trajectory", "sample_trajectory_embedded"]:
            if key in samples_dict and samples_dict[key] is not None:
                traj = samples_dict[key]
                break

        if traj is not None:
            traj = traj.detach().cpu().numpy()
            # Expected shape: [batch, samples, T, D]
            try:
                _, _, T, D = traj.shape
            except Exception:
                T = traj.shape[-2]
                D = traj.shape[-1]
            for t in range(T):
                vec = traj[0, 0, t, :neural_dim]
                labels.append(f"diff_{t}")
                states.append(vec)

        return labels, states


def bin_angle(angle, bin_size):
    return int(angle // bin_size) % int(360 // bin_size)


def angle_to_colour(bin_idx, n_bins):
    hue = bin_idx / n_bins
    return colorsys.hsv_to_rgb(hue, 0.9, 0.9)


def bin_by_target_and_distractor(states, metadata, cue, n_bins):
    bin_size = 360.0 / n_bins
    target_binned = {b: [] for b in range(n_bins)}
    distractor_binned = {b: [] for b in range(n_bins)}

    for state, trial_metadata in zip(states, metadata):
        _, c1, c2 = trial_metadata
        if cue == 1:
            target_angle = c1
            distractor_angle = c2
        else:
            target_angle = c2
            distractor_angle = c1

        target_binned[bin_angle(target_angle, bin_size)].append(state)
        distractor_binned[bin_angle(distractor_angle, bin_size)].append(state)

    target_avg = np.full((n_bins, states.shape[1]), np.nan, dtype=np.float32)
    distractor_avg = np.full((n_bins, states.shape[1]), np.nan, dtype=np.float32)
    target_counts = np.zeros(n_bins, dtype=int)
    distractor_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        if target_binned[b]:
            target_avg[b] = np.mean(target_binned[b], axis=0)
            target_counts[b] = len(target_binned[b])
        if distractor_binned[b]:
            distractor_avg[b] = np.mean(distractor_binned[b], axis=0)
            distractor_counts[b] = len(distractor_binned[b])

    return target_avg, distractor_avg, target_counts, distractor_counts


def fit_plane_to_data(data):
    center = np.mean(data, axis=0)
    centered = data - center
    pca = PCA(n_components=3)
    pca.fit(centered)

    normal = pca.components_[2]
    normal = normal / np.linalg.norm(normal)
    variance = pca.explained_variance_ratio_
    planarity = variance[:2].sum()
    return normal, center, planarity, variance


def angle_between_planes_degrees(normal1, normal2):
    cos_angle = np.abs(np.dot(normal1, normal2))
    cos_angle = min(1.0, max(0.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def create_plane_mesh(normal, center, size):
    if abs(normal[0]) < 0.9:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([0.0, 1.0, 0.0])
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)

    u = np.linspace(-size, size, 10)
    v = np.linspace(-size, size, 10)
    U, V = np.meshgrid(u, v)
    X = center[0] + U * v1[0] + V * v2[0]
    Y = center[1] + U * v1[1] + V * v2[1]
    Z = center[2] + U * v1[2] + V * v2[2]
    return X, Y, Z


def plot_plane_fitting_visualization(
    target_pca,
    distractor_pca,
    target_bins,
    distractor_bins,
    pca,
    plane_angle,
    out_path,
):
    combined = np.vstack([target_pca, distractor_pca])
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.12 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]
    plane_size = 0.85 * float(np.max(span))

    target_normal, target_center, target_planarity, _ = fit_plane_to_data(target_pca)
    distractor_normal, distractor_center, distractor_planarity, _ = fit_plane_to_data(
        distractor_pca
    )

    target_order = np.argsort(target_bins)
    distractor_order = np.argsort(distractor_bins)

    target_X, target_Y, target_Z = create_plane_mesh(target_normal, target_center, plane_size)
    distractor_X, distractor_Y, distractor_Z = create_plane_mesh(
        distractor_normal, distractor_center, plane_size
    )

    fig = plt.figure(figsize=(16, 12))
    fig.subplots_adjust(top=0.90)

    metrics_text = (
        f"Angle: {plane_angle:.2f} deg\n"
        f"Target planarity: {target_planarity:.1%}\n"
        f"Distractor planarity: {distractor_planarity:.1%}\n"
        f"Bins: {len(target_bins)} target / {len(distractor_bins)} distractor"
    )
    fig.text(
        0.52,
        0.988,
        metrics_text,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95),
    )

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot_surface(target_X, target_Y, target_Z, alpha=0.10, color="red", edgecolor="none")
    ax.plot_surface(distractor_X, distractor_Y, distractor_Z, alpha=0.10, color="blue", edgecolor="none")
    ax.plot(
        target_pca[target_order, 0],
        target_pca[target_order, 1],
        target_pca[target_order, 2],
        color="darkred",
        linewidth=2.2,
        alpha=0.75,
    )
    ax.plot(
        distractor_pca[distractor_order, 0],
        distractor_pca[distractor_order, 1],
        distractor_pca[distractor_order, 2],
        color="darkblue",
        linewidth=2.0,
        alpha=0.75,
        linestyle="--",
    )
    ax.scatter(
        target_pca[:, 0],
        target_pca[:, 1],
        target_pca[:, 2],
        c=[angle_to_colour(int(b), len(target_bins)) for b in target_bins],
        marker="o",
        s=100,
        edgecolors="darkred",
        linewidths=1.6,
        alpha=0.85,
        label="Target",
    )
    ax.scatter(
        distractor_pca[:, 0],
        distractor_pca[:, 1],
        distractor_pca[:, 2],
        c=[angle_to_colour(int(b), len(distractor_bins)) for b in distractor_bins],
        marker="^",
        s=100,
        edgecolors="darkblue",
        linewidths=1.5,
        alpha=0.85,
        label="Distractor",
    )
    arrow_scale = 0.35 * plane_size
    ax.quiver(
        0,
        0,
        0,
        target_normal[0] * arrow_scale,
        target_normal[1] * arrow_scale,
        target_normal[2] * arrow_scale,
        color="darkred",
        arrow_length_ratio=0.2,
        linewidth=2.5,
        label="Target normal",
    )
    ax.quiver(
        0,
        0,
        0,
        distractor_normal[0] * arrow_scale,
        distractor_normal[1] * arrow_scale,
        distractor_normal[2] * arrow_scale,
        color="darkblue",
        arrow_length_ratio=0.2,
        linewidth=2.5,
        label="Distractor normal",
    )
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12, fontweight="bold")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})", fontsize=12, fontweight="bold")
    ax.set_title(
        f"3D View with Fitted Planes\nAngle = {plane_angle:.2f} deg",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(
        target_pca[target_order, 0],
        target_pca[target_order, 1],
        color="darkred",
        linewidth=2.0,
        alpha=0.7,
    )
    ax2.plot(
        distractor_pca[distractor_order, 0],
        distractor_pca[distractor_order, 1],
        color="darkblue",
        linewidth=2.0,
        alpha=0.7,
        linestyle="--",
    )
    ax2.scatter(
        target_pca[:, 0],
        target_pca[:, 1],
        c=[angle_to_colour(int(b), len(target_bins)) for b in target_bins],
        marker="o",
        s=100,
        edgecolors="darkred",
        linewidths=1.6,
        alpha=0.85,
        label="Target",
    )
    ax2.scatter(
        distractor_pca[:, 0],
        distractor_pca[:, 1],
        c=[angle_to_colour(int(b), len(distractor_bins)) for b in distractor_bins],
        marker="^",
        s=100,
        edgecolors="darkblue",
        linewidths=1.5,
        alpha=0.85,
        label="Distractor",
    )
    target_normal_proj = target_normal[:2] / max(np.linalg.norm(target_normal[:2]), 1e-12)
    distractor_normal_proj = distractor_normal[:2] / max(np.linalg.norm(distractor_normal[:2]), 1e-12)
    target_line_dir = np.array([-target_normal_proj[1], target_normal_proj[0]])
    distractor_line_dir = np.array([-distractor_normal_proj[1], distractor_normal_proj[0]])
    line_length = 1.15 * np.max(span[:2])
    for sign in [-1, 1]:
        ax2.plot(
            [0, sign * line_length * target_line_dir[0]],
            [0, sign * line_length * target_line_dir[1]],
            "r--",
            linewidth=2,
            alpha=0.5,
        )
        ax2.plot(
            [0, sign * line_length * distractor_line_dir[0]],
            [0, sign * line_length * distractor_line_dir[1]],
            "b--",
            linewidth=2,
            alpha=0.5,
        )
    ax2.set_xlim(lims[0])
    ax2.set_ylim(lims[1])
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12, fontweight="bold")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12, fontweight="bold")
    ax2.set_title("PC1-PC2 View", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(
        target_pca[target_order, 0],
        target_pca[target_order, 2],
        color="darkred",
        linewidth=2.0,
        alpha=0.7,
    )
    ax3.plot(
        distractor_pca[distractor_order, 0],
        distractor_pca[distractor_order, 2],
        color="darkblue",
        linewidth=2.0,
        alpha=0.7,
        linestyle="--",
    )
    ax3.scatter(
        target_pca[:, 0],
        target_pca[:, 2],
        c=[angle_to_colour(int(b), len(target_bins)) for b in target_bins],
        marker="o",
        s=100,
        edgecolors="darkred",
        linewidths=1.6,
        alpha=0.85,
        label="Target",
    )
    ax3.scatter(
        distractor_pca[:, 0],
        distractor_pca[:, 2],
        c=[angle_to_colour(int(b), len(distractor_bins)) for b in distractor_bins],
        marker="^",
        s=100,
        edgecolors="darkblue",
        linewidths=1.5,
        alpha=0.85,
        label="Distractor",
    )
    ax3.set_xlim(lims[0])
    ax3.set_ylim(lims[2])
    ax3.set_xlabel("PC1", fontsize=12, fontweight="bold")
    ax3.set_ylabel("PC3", fontsize=12, fontweight="bold")
    ax3.set_title("PC1-PC3 View", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(
        target_pca[target_order, 1],
        target_pca[target_order, 2],
        color="darkred",
        linewidth=2.0,
        alpha=0.7,
    )
    ax4.plot(
        distractor_pca[distractor_order, 1],
        distractor_pca[distractor_order, 2],
        color="darkblue",
        linewidth=2.0,
        alpha=0.7,
        linestyle="--",
    )
    ax4.scatter(
        target_pca[:, 1],
        target_pca[:, 2],
        c=[angle_to_colour(int(b), len(target_bins)) for b in target_bins],
        marker="o",
        s=100,
        edgecolors="darkred",
        linewidths=1.6,
        alpha=0.85,
        label="Target",
    )
    ax4.scatter(
        distractor_pca[:, 1],
        distractor_pca[:, 2],
        c=[angle_to_colour(int(b), len(distractor_bins)) for b in distractor_bins],
        marker="^",
        s=100,
        edgecolors="darkblue",
        linewidths=1.5,
        alpha=0.85,
        label="Distractor",
    )
    ax4.set_xlim(lims[1])
    ax4.set_ylim(lims[2])
    ax4.set_xlabel("PC2", fontsize=12, fontweight="bold")
    ax4.set_ylabel("PC3", fontsize=12, fontweight="bold")
    ax4.set_title("PC2-PC3 View", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "target_planarity": float(target_planarity),
        "distractor_planarity": float(distractor_planarity),
        "target_normal": target_normal.tolist(),
        "distractor_normal": distractor_normal.tolist(),
        "target_center": target_center.tolist(),
        "distractor_center": distractor_center.tolist(),
    }


def main():
    args = parse_args()

    prep_indices_to_analyze = list(range(4)) if args.all_prep_indices else [args.prep_idx]

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
            / "single_cue_target_distractor_planes"
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

    trials = generate_trial_combinations(args.cue, args.angle_step)
    print(f"Generated {len(trials)} trials for cue={args.cue}")
    if len(trials) != 144:
        print(f"Warning: expected 144 trials, got {len(trials)}")

    # Two modes: per-prep-index analysis (default / --all-prep-indices)
    # or full time-series analysis (--all-timesteps)
    if args.all_timesteps:
        timepoint_order = None
        timepoint_states = {}
        metadata = []

        for i, trial in enumerate(trials):
            if i > 0 and i % 24 == 0:
                print(f"  trial progress: {i}/{len(trials)}")
            labels, states_list = extract_all_states_for_trial(
                trial=trial, task=task, model=model, device=device, neural_dim=args.neural_dim
            )
            if timepoint_order is None:
                timepoint_order = list(labels)
                for lbl in timepoint_order:
                    timepoint_states[lbl] = []
            else:
                if list(labels) != timepoint_order:
                    print("Warning: inconsistent timepoint labels across trials")

            for lbl, st in zip(labels, states_list):
                timepoint_states[lbl].append(st)
            metadata.append([trial["cue"], trial["target_angle"], trial["distractor_angle"]])

        metadata = np.asarray(metadata, dtype=float)
        all_results = {}

        for tp in timepoint_order:
            print(f"\nAnalyzing timepoint={tp}...")
            states_tp = np.asarray(timepoint_states[tp])

            target_avg, distractor_avg, target_counts, distractor_counts = bin_by_target_and_distractor(
                states=states_tp, metadata=metadata, cue=args.cue, n_bins=args.n_bins
            )

            target_valid = ~np.isnan(target_avg).any(axis=1)
            distractor_valid = ~np.isnan(distractor_avg).any(axis=1)
            target_avg = target_avg[target_valid]
            distractor_avg = distractor_avg[distractor_valid]
            target_bins = np.where(target_valid)[0]
            distractor_bins = np.where(distractor_valid)[0]

            combined_states = np.vstack([target_avg, distractor_avg])
            combined_meta = np.array(
                [(0, int(b)) for b in target_bins] + [(1, int(b)) for b in distractor_bins],
                dtype=int,
            )

            pca = PCA(n_components=3)
            combined_pca = pca.fit_transform(combined_states)
            target_pca = combined_pca[combined_meta[:, 0] == 0]
            distractor_pca = combined_pca[combined_meta[:, 0] == 1]

            target_normal, target_center, target_planarity, _ = fit_plane_to_data(target_pca)
            distractor_normal, distractor_center, distractor_planarity, _ = fit_plane_to_data(
                distractor_pca
            )
            plane_angle = angle_between_planes_degrees(target_normal, distractor_normal)
            plane_sep = float(np.linalg.norm(target_center - distractor_center))

            print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
            print(f"  Angle: {plane_angle:.2f} deg")
            print(f"  Target planarity: {target_planarity:.2%}, Distractor planarity: {distractor_planarity:.2%}")

            figure_path = output_dir / f"single_cue_target_distractor_planes_{tp}.png"
            plot_info = plot_plane_fitting_visualization(
                target_pca=target_pca,
                distractor_pca=distractor_pca,
                target_bins=target_bins,
                distractor_bins=distractor_bins,
                pca=pca,
                plane_angle=plane_angle,
                out_path=figure_path,
            )

            tp_summary = {
                "timepoint": tp,
                "combined_pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "plane_angle_deg": float(plane_angle),
                "plane_separation": float(plane_sep),
                "target_planarity": float(target_planarity),
                "distractor_planarity": float(distractor_planarity),
                "target_counts_per_bin": target_counts.tolist(),
                "distractor_counts_per_bin": distractor_counts.tolist(),
                "figure_path": str(figure_path),
            }
            tp_summary.update(plot_info)
            all_results[tp] = tp_summary

            tp_summary_path = output_dir / f"single_cue_target_distractor_planes_summary_{tp}.json"
            with open(tp_summary_path, "w", encoding="utf-8") as f:
                json.dump(tp_summary, f, indent=2)

        print("Saved outputs:")
        for tp in timepoint_order:
            figure_path = output_dir / f"single_cue_target_distractor_planes_{tp}.png"
            summary_path = output_dir / f"single_cue_target_distractor_planes_summary_{tp}.json"
            print(f"  {tp}: {figure_path}")
            print(f"     {summary_path}")

        master_summary = {
            "run_dir": str(run_dir),
            "cue": int(args.cue),
            "n_trials": int(len(trials)),
            "angle_step": int(args.angle_step),
            "n_bins": int(args.n_bins),
            "neural_dim": int(args.neural_dim),
            "timepoints_analyzed": list(timepoint_order),
            "per_timepoint_results": all_results,
        }

        with open(output_dir / "single_cue_target_distractor_planes_master_summary.json", "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2)
        print(f"\nMaster summary: {output_dir / 'single_cue_target_distractor_planes_master_summary.json'}")

    else:
        states = []
        metadata = []
        for i, trial in enumerate(trials):
            if i > 0 and i % 24 == 0:
                print(f"  trial progress: {i}/{len(trials)}")
            state = extract_state_for_trial(
                trial=trial,
                task=task,
                model=model,
                device=device,
                prep_idx=max(prep_indices_to_analyze),
                neural_dim=args.neural_dim,
            )
            states.append(state)
            metadata.append([trial["cue"], trial["target_angle"], trial["distractor_angle"]])

        states = np.asarray(states)
        metadata = np.asarray(metadata, dtype=float)

        all_results = {}

        for prep_idx in prep_indices_to_analyze:
            print(f"\nAnalyzing prep_idx={prep_idx}...")

            target_avg, distractor_avg, target_counts, distractor_counts = bin_by_target_and_distractor(
                states=states,
                metadata=metadata,
                cue=args.cue,
                n_bins=args.n_bins,
            )

            target_valid = ~np.isnan(target_avg).any(axis=1)
            distractor_valid = ~np.isnan(distractor_avg).any(axis=1)
            target_avg = target_avg[target_valid]
            distractor_avg = distractor_avg[distractor_valid]
            target_bins = np.where(target_valid)[0]
            distractor_bins = np.where(distractor_valid)[0]

            combined_states = np.vstack([target_avg, distractor_avg])
            combined_meta = np.array(
                [(0, int(b)) for b in target_bins] + [(1, int(b)) for b in distractor_bins],
                dtype=int,
            )

            pca = PCA(n_components=3)
            combined_pca = pca.fit_transform(combined_states)
            target_pca = combined_pca[combined_meta[:, 0] == 0]
            distractor_pca = combined_pca[combined_meta[:, 0] == 1]

            target_normal, target_center, target_planarity, _ = fit_plane_to_data(target_pca)
            distractor_normal, distractor_center, distractor_planarity, _ = fit_plane_to_data(
                distractor_pca
            )
            plane_angle = angle_between_planes_degrees(target_normal, distractor_normal)
            plane_sep = float(np.linalg.norm(target_center - distractor_center))

            print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
            print(f"  Angle: {plane_angle:.2f} deg")
            print(f"  Target planarity: {target_planarity:.2%}, Distractor planarity: {distractor_planarity:.2%}")

            figure_path = output_dir / f"single_cue_target_distractor_planes_prep_{prep_idx}.png"
            plot_info = plot_plane_fitting_visualization(
                target_pca=target_pca,
                distractor_pca=distractor_pca,
                target_bins=target_bins,
                distractor_bins=distractor_bins,
                pca=pca,
                plane_angle=plane_angle,
                out_path=figure_path,
            )

            prep_summary = {
                "prep_idx": int(prep_idx),
                "combined_pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "plane_angle_deg": float(plane_angle),
                "plane_separation": float(plane_sep),
                "target_planarity": float(target_planarity),
                "distractor_planarity": float(distractor_planarity),
                "target_counts_per_bin": target_counts.tolist(),
                "distractor_counts_per_bin": distractor_counts.tolist(),
                "figure_path": str(figure_path),
            }
            prep_summary.update(plot_info)
            all_results[int(prep_idx)] = prep_summary

            prep_summary_path = output_dir / f"single_cue_target_distractor_planes_summary_prep_{prep_idx}.json"
            with open(prep_summary_path, "w", encoding="utf-8") as f:
                json.dump(prep_summary, f, indent=2)

        print("Saved outputs:")
        for prep_idx in prep_indices_to_analyze:
            figure_path = output_dir / f"single_cue_target_distractor_planes_prep_{prep_idx}.png"
            summary_path = output_dir / f"single_cue_target_distractor_planes_summary_prep_{prep_idx}.json"
            print(f"  prep_idx={prep_idx}: {figure_path}")
            print(f"               {summary_path}")

        master_summary = {
            "run_dir": str(run_dir),
            "cue": int(args.cue),
            "n_trials": int(len(trials)),
            "angle_step": int(args.angle_step),
            "n_bins": int(args.n_bins),
            "neural_dim": int(args.neural_dim),
            "prep_indices_analyzed": [int(p) for p in prep_indices_to_analyze],
            "per_prep_results": all_results,
        }

        with open(output_dir / "single_cue_target_distractor_planes_master_summary.json", "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2)
        print(f"\nMaster summary: {output_dir / 'single_cue_target_distractor_planes_master_summary.json'}")


if __name__ == "__main__":
    main()