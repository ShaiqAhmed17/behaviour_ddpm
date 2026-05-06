"""
Standalone prospective-memory geometry analysis with a single global PCA.

This script separates the shared-coordinate analysis from the notebook and makes
it reproducible from the command line.

What it does:
1. Loads a DDPM model checkpoint.
2. Generates all non-swap trials over cue and color combinations.
3. Optionally sweeps ablation directions for each cue/color trial.
4. Extracts neural states for multiple prep indices.
4. Bins states by a shared feature (cued-color bin).
5. Fits one global PCA across all prep indices.
6. Visualizes each prep index in the same PCA coordinate system.

Example:
    conda activate ddpm
    python analyze_ablated_teacher_trajectories.py \
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
        description="Consistent PCA across prep states using common cued-color binning."
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
        help="Path relative to repo-root for model run directory containing args.yaml and state.mdl.",
    )
    parser.add_argument(
        "--prep-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Prep indices to analyze in a shared PCA basis.",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=30,
        help="Color angle step in degrees.",
    )
    parser.add_argument(
        "--neural-dim",
        type=int,
        default=16,
        help="Number of neural dimensions to extract from postprep_state.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=12,
        help="Number of cued-color bins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ddpm/analysis/new_analysis/results/<run_name>/global_pca.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device to use.",
    )
    parser.add_argument(
        "--ablation-directions",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Ablation direction indices from model.behaviour_nullspace to analyze. "
            "If omitted and --all-ablation-directions is not set, no ablation is applied."
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
        help="Also include the healthy (no-ablation) condition alongside ablated directions.",
    )
    return parser.parse_args()


def generate_trial_combinations(angle_step, ablation_directions):
    angles = list(range(0, 360, angle_step))
    trials = []
    for ablation_direction in ablation_directions:
        for cue in [1, 2]:
            for color1 in angles:
                for color2 in angles:
                    trials.append(
                        {
                            "cue": cue,
                            "color1_angle": color1,
                            "color2_angle": color2,
                            "swap": False,
                            "ablation_direction": int(ablation_direction),
                        }
                    )
    return trials


def extract_neural_state_from_model(
    trial,
    task,
    model,
    device,
    prep_idx,
    neural_dim,
    ablation_vectors,
):
    with torch.no_grad():
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

        trial_info = task.generate_trial_information(
            batch_size=1,
            num_samples=1,
            override_task_variable_information=task_variable_dict,
        )

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

        ablation_direction = int(trial["ablation_direction"])
        if ablation_direction >= 0:
            sample_kwargs["ablation_vector"] = ablation_vectors[ablation_direction]

        prep_dicts, _ = model.generate_samples(**sample_kwargs)

        if prep_idx >= len(prep_dicts):
            raise ValueError(
                f"prep_idx {prep_idx} exceeds number of prep epochs {len(prep_dicts)}"
            )

        return prep_dicts[prep_idx]["postprep_state"][0, 0, :neural_dim].cpu().numpy()


def bin_angle(angle, bin_size):
    return int(angle // bin_size) % int(360 // bin_size)


def bin_and_average_by_cued_color(states, metadata, n_bins):
    bin_size = 360.0 / n_bins
    binned_data = {
        1: {b: [] for b in range(n_bins)},
        2: {b: [] for b in range(n_bins)},
    }

    for i, trial_metadata in enumerate(metadata):
        cue, c1, c2 = trial_metadata[:3]
        cue = int(cue)
        cued_angle = c1 if cue == 1 else c2
        bin_idx = bin_angle(cued_angle, bin_size)
        binned_data[cue][bin_idx].append(states[i])

    averaged_data = {
        1: np.full((n_bins, states.shape[1]), np.nan),
        2: np.full((n_bins, states.shape[1]), np.nan),
    }

    for cue in [1, 2]:
        for b in range(n_bins):
            if binned_data[cue][b]:
                averaged_data[cue][b] = np.mean(binned_data[cue][b], axis=0)

    return averaged_data, binned_data


def angle_to_colour(bin_idx, n_bins):
    hue = bin_idx / n_bins
    return colorsys.hsv_to_rgb(hue, 0.9, 0.9)


def get_states_for_prep(trials, prep_idx, task, model, device, neural_dim, ablation_vectors):
    states = []
    metadata = []

    print(f"Extracting states for prep_idx={prep_idx}...")
    for i, trial in enumerate(trials):
        if i % 200 == 0 and i > 0:
            print(f"  progress: {i}/{len(trials)}")
        state = extract_neural_state_from_model(
            trial=trial,
            task=task,
            model=model,
            device=device,
            prep_idx=prep_idx,
            neural_dim=neural_dim,
            ablation_vectors=ablation_vectors,
        )
        states.append(state)
        metadata.append(
            [
                trial["cue"],
                trial["color1_angle"],
                trial["color2_angle"],
                trial["ablation_direction"],
            ]
        )

    return np.array(states), np.array(metadata)


def get_requested_ablation_directions(args, model):
    if args.ablation_directions is not None and args.all_ablation_directions:
        raise ValueError("Use either --ablation-directions or --all-ablation-directions, not both.")

    if args.all_ablation_directions:
        max_dirs = model.behaviour_nullspace.shape[0]
        ablation_directions = list(range(max_dirs))
    elif args.ablation_directions is not None:
        ablation_directions = sorted(set(int(d) for d in args.ablation_directions))
        max_dirs = model.behaviour_nullspace.shape[0]
        bad = [d for d in ablation_directions if d < 0 or d >= max_dirs]
        if bad:
            raise ValueError(
                f"Invalid ablation direction(s) {bad}; valid range is [0, {max_dirs - 1}]"
            )
    else:
        ablation_directions = []

    if args.include_healthy or not ablation_directions:
        return [-1] + ablation_directions
    return ablation_directions


def build_ablation_vectors(model, direction_indices, device):
    ablation_vectors = {}
    for direction_idx in direction_indices:
        vec = model.behaviour_nullspace[direction_idx].clone()
        vec = vec / torch.norm(vec)
        ablation_vectors[int(direction_idx)] = vec.to(device)
    return ablation_vectors


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


def make_global_pca_figure(pooled_pca, pooled_labels, prep_indices, prep_names, pca_global, out_path, n_bins):
    mins = pooled_pca.min(axis=0)
    maxs = pooled_pca.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    lims = [(mins[d] - pad[d], maxs[d] + pad[d]) for d in range(3)]

    fig = plt.figure(figsize=(18, 20))
    global_frac = pca_global.explained_variance_ratio_

    for row, prep_idx in enumerate(prep_indices):
        prep_mask = pooled_labels[:, 0] == prep_idx
        coords = pooled_pca[prep_mask]
        cues = pooled_labels[prep_mask, 1]
        bins = pooled_labels[prep_mask, 2]
        if coords.shape[0] > 1:
            local_var = coords.var(axis=0, ddof=1)
            local_frac = local_var / max(local_var.sum(), 1e-12)
        else:
            local_frac = np.zeros(3)

        ax3d = fig.add_subplot(len(prep_indices), 3, row * 3 + 1, projection="3d")
        for cue_val, marker in [(1, "o"), (2, "^")]:
            mask = cues == cue_val
            ax3d.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=[angle_to_colour(int(b), n_bins) for b in bins[mask]],
                marker=marker,
                s=90,
                edgecolors="k",
                linewidths=0.8,
                alpha=0.9,
            )
        ax3d.set_xlim(lims[0])
        ax3d.set_ylim(lims[1])
        ax3d.set_zlim(lims[2])
        ax3d.set_xlabel(f"PC1 (g:{global_frac[0]:.1%}, l:{local_frac[0]:.1%})")
        ax3d.set_ylabel(f"PC2 (g:{global_frac[1]:.1%}, l:{local_frac[1]:.1%})")
        ax3d.set_zlabel(f"PC3 (g:{global_frac[2]:.1%}, l:{local_frac[2]:.1%})")
        ax3d.set_title(
            (
                f"prep_idx={prep_idx}: {prep_names.get(prep_idx, str(prep_idx))}\n"
                f"local var frac: PC1={local_frac[0]:.1%}, PC2={local_frac[1]:.1%}, PC3={local_frac[2]:.1%}"
            ),
            fontweight="bold",
        )
        ax3d.grid(True, alpha=0.25)

        marker_legend = [
            Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 1"),
            Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=7, label="Cue 2"),
        ]
        ax3d.legend(handles=marker_legend, fontsize=8, loc="upper right")

        ax12 = fig.add_subplot(len(prep_indices), 3, row * 3 + 2)
        for cue_val, marker in [(1, "o"), (2, "^")]:
            mask = cues == cue_val
            ax12.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[angle_to_colour(int(b), n_bins) for b in bins[mask]],
                marker=marker,
                s=90,
                edgecolors="k",
                linewidths=0.8,
                alpha=0.9,
            )
        ax12.set_xlim(lims[0])
        ax12.set_ylim(lims[1])
        ax12.set_xlabel(f"PC1 (g:{global_frac[0]:.1%}, l:{local_frac[0]:.1%})", fontweight="bold")
        ax12.set_ylabel(f"PC2 (g:{global_frac[1]:.1%}, l:{local_frac[1]:.1%})", fontweight="bold")
        ax12.set_title(
            f"PC1 vs PC2 (local: {local_frac[0]:.1%}, {local_frac[1]:.1%})",
            fontweight="bold",
        )
        ax12.grid(True, alpha=0.25)
        ax12.set_aspect("equal", adjustable="box")

        ax23 = fig.add_subplot(len(prep_indices), 3, row * 3 + 3)
        for cue_val, marker in [(1, "o"), (2, "^")]:
            mask = cues == cue_val
            ax23.scatter(
                coords[mask, 1],
                coords[mask, 2],
                c=[angle_to_colour(int(b), n_bins) for b in bins[mask]],
                marker=marker,
                s=90,
                edgecolors="k",
                linewidths=0.8,
                alpha=0.9,
            )
        ax23.set_xlim(lims[1])
        ax23.set_ylim(lims[2])
        ax23.set_xlabel(f"PC2 (g:{global_frac[1]:.1%}, l:{local_frac[1]:.1%})", fontweight="bold")
        ax23.set_ylabel(f"PC3 (g:{global_frac[2]:.1%}, l:{local_frac[2]:.1%})", fontweight="bold")
        ax23.set_title(
            f"PC2 vs PC3 (local: {local_frac[1]:.1%}, {local_frac[2]:.1%})",
            fontweight="bold",
        )
        ax23.grid(True, alpha=0.25)
        ax23.set_aspect("equal", adjustable="box")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
            / "global_pca"
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
    print(f"Args: {args_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(args_path), device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    all_ablation_directions = get_requested_ablation_directions(args, model)
    active_ablation_directions = [d for d in all_ablation_directions if d >= 0]
    ablation_vectors = build_ablation_vectors(model, active_ablation_directions, device)

    trials = generate_trial_combinations(
        angle_step=args.angle_step,
        ablation_directions=all_ablation_directions,
    )
    print(
        f"Generated {len(trials)} trials "
        f"({len(all_ablation_directions)} ablation conditions: {all_ablation_directions})"
    )

    prep_names = {
        0: "Cue presentation",
        1: "Delay 1 onset",
        2: "Delay 2 onset",
        3: "Pre-response",
    }

    pooled_states = []
    pooled_labels = []
    per_prep_separation = {}
    per_prep_local_variance_fraction = {}

    for prep_idx in args.prep_indices:
        states, metadata = get_states_for_prep(
            trials=trials,
            prep_idx=prep_idx,
            task=task,
            model=model,
            device=device,
            neural_dim=args.neural_dim,
            ablation_vectors=ablation_vectors,
        )

        direction_values = metadata[:, 3]
        for ablation_direction in np.unique(direction_values):
            direction_mask = direction_values == ablation_direction
            direction_averaged, _ = bin_and_average_by_cued_color(
                states=states[direction_mask],
                metadata=metadata[direction_mask],
                n_bins=args.n_bins,
            )
            for cue in [1, 2]:
                for b in range(args.n_bins):
                    direction_vec = direction_averaged[cue][b]
                    if np.isnan(direction_vec).any():
                        continue
                    pooled_states.append(direction_vec)
                    pooled_labels.append((prep_idx, cue, b, int(ablation_direction)))

    pooled_states = np.array(pooled_states)
    pooled_labels = np.array(pooled_labels, dtype=int)

    pca_global = PCA(n_components=3)
    pooled_pca = pca_global.fit_transform(pooled_states)

    print("Global PCA explained variance ratio:", pca_global.explained_variance_ratio_)
    print("Global PCA cumulative variance:", pca_global.explained_variance_ratio_.cumsum())

    for prep_idx in args.prep_indices:
        pmask = pooled_labels[:, 0] == prep_idx
        coords = pooled_pca[pmask]
        cues = pooled_labels[pmask, 1]
        c1_center = coords[cues == 1].mean(axis=0)
        c2_center = coords[cues == 2].mean(axis=0)
        per_prep_separation[int(prep_idx)] = float(np.linalg.norm(c1_center - c2_center))
        if coords.shape[0] > 1:
            var = coords.var(axis=0, ddof=1)
            frac = var / max(var.sum(), 1e-12)
            per_prep_local_variance_fraction[int(prep_idx)] = {
                "PC1": float(frac[0]),
                "PC2": float(frac[1]),
                "PC3": float(frac[2]),
            }
        else:
            per_prep_local_variance_fraction[int(prep_idx)] = {
                "PC1": 0.0,
                "PC2": 0.0,
                "PC3": 0.0,
            }

    figure_path = output_dir / "prospective_memory_global_pca_common_cued_bin.png"
    make_global_pca_figure(
        pooled_pca=pooled_pca,
        pooled_labels=pooled_labels,
        prep_indices=args.prep_indices,
        prep_names=prep_names,
        pca_global=pca_global,
        out_path=figure_path,
        n_bins=args.n_bins,
    )

    np.savez(
        output_dir / "global_pca_data.npz",
        pooled_states=pooled_states,
        pooled_labels=pooled_labels,
        pooled_pca=pooled_pca,
        explained_variance=pca_global.explained_variance_ratio_,
        pca_components=pca_global.components_,
        pca_mean=pca_global.mean_,
    )

    summary = {
        "run_dir": str(run_dir),
        "prep_indices": [int(p) for p in args.prep_indices],
        "n_trials": len(trials),
        "ablation_directions": [int(d) for d in all_ablation_directions],
        "healthy_direction_code": -1,
        "n_bins": int(args.n_bins),
        "angle_step": int(args.angle_step),
        "neural_dim": int(args.neural_dim),
        "explained_variance_ratio": pca_global.explained_variance_ratio_.tolist(),
        "explained_variance_cumulative": pca_global.explained_variance_ratio_.cumsum().tolist(),
        "per_prep_cue_center_separation": per_prep_separation,
        "per_prep_local_variance_fraction_in_global_basis": per_prep_local_variance_fraction,
        "figure_path": str(figure_path),
    }

    with open(output_dir / "global_pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs:")
    print(f"  {figure_path}")
    print(f"  {output_dir / 'global_pca_data.npz'}")
    print(f"  {output_dir / 'global_pca_summary.json'}")


if __name__ == "__main__":
    main()
