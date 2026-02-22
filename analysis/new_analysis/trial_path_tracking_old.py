import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch


def _parse_prep_indices(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_trials(angle_step, max_trials, seed):
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for color1 in angles:
            for color2 in angles:
                trials.append(
                    {
                        "cue": cue,
                        "color1_angle": float(color1),
                        "color2_angle": float(color2),
                    }
                )

    rng = np.random.default_rng(seed)
    rng.shuffle(trials)
    if max_trials is not None and max_trials > 0:
        trials = trials[: max_trials]
    return trials


def _trial_to_variable_dict(trial, task):
    probe_features = torch.tensor([[trial["color1_angle"], trial["color2_angle"]]]) * (
        np.pi / 180.0
    )
    report_features = probe_features.clone()

    override_stimulus_features = {
        "probe_features": probe_features,
        "report_features": report_features,
    }
    override_stimulus_cart_features = {}
    for key in override_stimulus_features.keys():
        override_stimulus_cart_features[f"{key}_cart"] = torch.stack(
            polar2cart(1.0, override_stimulus_features[key]), -1
        )

    override_stimulus_features_dict = dict(
        **override_stimulus_features, **override_stimulus_cart_features
    )
    override_stimulus_features_dict["cued_item_idx"] = torch.tensor(
        [trial["cue"] - 1]
    )

    return task.task_variable_gen.generate_variable_dict(
        batch_size=1,
        override_stimulus_features_dict=override_stimulus_features_dict,
    )


def _to_device_inputs(inputs, device):
    moved = []
    for inp in inputs:
        moved.append(inp.to(device) if isinstance(inp, torch.Tensor) else inp)
    return moved


def _is_non_swap_trial(samples, report_features_cart, cued_item_idx, sample_radius, p_thres):
    if report_features_cart.ndim == 2:
        report_features_cart = report_features_cart.unsqueeze(0)

    if samples.ndim == 2:
        samples = samples.unsqueeze(0).unsqueeze(1)
    elif samples.ndim == 3:
        samples = samples.unsqueeze(1)

    report_features_cart = report_features_cart.to(samples.device)
    cued_item_idx = cued_item_idx.to(samples.device)

    # samples: [B, S, 1, 2] -> [B, S, items, 2] after broadcast
    diff = (samples / sample_radius) - report_features_cart[:, None, :, :]
    square_errors = diff.square().sum(-1)
    p_components = (-square_errors).softmax(-1)

    gather_idx = cued_item_idx[:, None, None].expand(
        p_components.shape[0], p_components.shape[1], 1
    )
    p_correct = p_components.gather(-1, gather_idx).squeeze(-1)
    return p_correct.mean(1) >= p_thres


def _collect_epoch_trajectories(prep_dicts, samples_dict, prep_indices):
    trajectories = {}
    for prep_idx in prep_indices:
        if prep_idx >= len(prep_dicts):
            continue
        traj = prep_dicts[prep_idx]["preparatory_trajectory"].mean(1)
        trajectories[f"prep_{prep_idx}"] = traj

    if "embedded_sample_trajectory" in samples_dict:
        trajectories["diffusion"] = samples_dict["embedded_sample_trajectory"].mean(1)

    return trajectories


def _angle_to_color(angle_radians):
    """
    Convert angle in radians to HSV color.
    Maps 0 to 2*pi to full hue range.
    """
    hue = (angle_radians % (2 * np.pi)) / (2 * np.pi)
    return colorsys.hsv_to_rgb(hue, 0.9, 0.9)


def _plot_epoch_trajectories(epoch_trajs, trial_metadata, out_path, title, epoch_name, color_by_stimulus):
    """
    Plot trajectories in 3-component PCA space with multiple 2D views.
    
    Args:
        epoch_trajs: List of trajectory arrays [time, features]
        trial_metadata: List of dicts with 'cue', 'color1_angle', 'color2_angle'
        out_path: Path to save figure
        title: Plot title
        epoch_name: Name of epoch (e.g., 'prep_2', 'diffusion')
        color_by_stimulus: If True, color by stimulus angle; if False, color by cue
    """
    if len(epoch_trajs) == 0:
        return

    all_points = np.concatenate(epoch_trajs, axis=0)
    pca = PCA(n_components=3)
    proj = pca.fit_transform(all_points)

    lengths = [traj.shape[0] for traj in epoch_trajs]
    splits = np.cumsum(lengths)[:-1]
    proj_trajs = np.split(proj, splits)

    # Create figure with 3 subplots for different PC pairs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Add main title with explained variance
    var_str = f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}"
    fig.suptitle(f"{title}\n{var_str}", fontsize=12, fontweight='bold')

    # Define PC pairs to plot
    pc_pairs = [(0, 1), (0, 2), (1, 2)]
    pc_labels = [
        (f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"),
        (f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"),
        (f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ]

    for ax_idx, (pc_pair, labels) in enumerate(zip(pc_pairs, pc_labels)):
        ax = axes[ax_idx]
        
        for traj_idx, traj in enumerate(proj_trajs):
            metadata = trial_metadata[traj_idx]
            
            # Determine color and marker
            if color_by_stimulus:
                # Color by cued stimulus angle
                cued_angle_degrees = metadata[f"color{metadata['cue']}_angle"]
                cued_angle_radians = cued_angle_degrees * np.pi / 180.0
                color = _angle_to_color(cued_angle_radians)
            else:
                # Color black, differentiate by marker
                color = 'black'
            
            # Marker by cue
            marker = 'o' if metadata['cue'] == 1 else '^'
            
            # Plot trajectory
            ax.plot(traj[:, pc_pair[0]], traj[:, pc_pair[1]], 
                   color=color if color_by_stimulus else 'black',
                   alpha=0.15, linewidth=0.8)
            ax.scatter(traj[:, pc_pair[0]], traj[:, pc_pair[1]], 
                      c=[color] if color_by_stimulus else ['black'],
                      marker=marker, s=20, alpha=0.6, edgecolors='none')
        
        ax.set_xlabel(labels[0], fontweight='bold')
        ax.set_ylabel(labels[1], fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return pca.explained_variance_ratio_


def main():
    parser = argparse.ArgumentParser(
        description="Track non-swap trial trajectories across task epochs."
    )
    parser.add_argument("--args-path", required=True, help="Path to args.yaml")
    parser.add_argument(
        "--checkpoint-path", required=True, help="Path to model checkpoint state.mdl"
    )
    parser.add_argument(
        "--results-dir",
        default="analysis/new_analysis/results",
        help="Output directory",
    )
    parser.add_argument("--angle-step", type=int, default=30)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--prep-indices", default="0,1,2,3")
    parser.add_argument("--p-decision-thres", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    args_path = Path(args.args_path)
    run_name = args_path.parent.name

    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
        str(args_path), device
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    trials = _build_trials(args.angle_step, args.max_trials, args.seed)
    prep_indices = _parse_prep_indices(args.prep_indices)

    epoch_to_trajs = {f"prep_{idx}": [] for idx in prep_indices}
    epoch_to_trajs["diffusion"] = []
    trial_metadata_per_epoch = {f"prep_{idx}": [] for idx in prep_indices}
    trial_metadata_per_epoch["diffusion"] = []

    total_trials = 0
    accepted_trials = 0

    for trial in trials:
        total_trials += 1
        task_variable_dict = _trial_to_variable_dict(trial, task)
        trial_info = task.generate_trial_information(
            batch_size=1,
            num_samples=args.num_samples,
            override_task_variable_information=task_variable_dict,
        )

        prep_inputs = _to_device_inputs(trial_info.prep_network_inputs, device)
        diff_inputs = _to_device_inputs(trial_info.diffusion_network_inputs, device)

        with torch.no_grad():
            prep_dicts, samples_dict = model.generate_samples(
                prep_network_inputs=prep_inputs,
                diffusion_network_inputs=diff_inputs,
                prep_epoch_durations=trial_info.prep_epoch_durations,
                diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
                samples_shape=[1, args.num_samples],
                noise_scaler=1.0,
            )

        samples = samples_dict["samples"]
        cued_idx = trial_info.task_variable_information["cued_item_idx"]
        report_features_cart = trial_info.task_variable_information["report_features_cart"]

        non_swap_mask = _is_non_swap_trial(
            samples,
            report_features_cart,
            cued_idx,
            task.sample_gen.sample_radius,
            args.p_decision_thres,
        )

        if not non_swap_mask.item():
            continue

        accepted_trials += 1
        epoch_trajs = _collect_epoch_trajectories(prep_dicts, samples_dict, prep_indices)
        for key, traj in epoch_trajs.items():
            epoch_to_trajs[key].append(traj.squeeze(0).cpu().numpy())
            trial_metadata_per_epoch[key].append(trial)

    epoch_to_trajs = {
        key: value for key, value in epoch_to_trajs.items() if len(value) > 0
    }
    trial_metadata_per_epoch = {
        key: value for key, value in trial_metadata_per_epoch.items() if len(value) > 0
    }

    variance_info = {}
    for epoch_name, trajs in epoch_to_trajs.items():
        out_path = results_dir / f"trial_paths_{run_name}_{epoch_name}.png"
        metadata = trial_metadata_per_epoch[epoch_name]
        
        # Determine if we should color by stimulus
        # Color by stimulus for prep_2, prep_3 and later (post-stimulus epochs)
        epoch_num = int(epoch_name.split('_')[1]) if 'prep' in epoch_name else None
        color_by_stimulus = (epoch_num is not None and epoch_num >= 2) or epoch_name == 'diffusion'
        
        explained_var = _plot_epoch_trajectories(
            trajs, metadata, out_path, 
            f"{run_name} | {epoch_name}", 
            epoch_name,
            color_by_stimulus
        )
        variance_info[epoch_name] = explained_var.tolist()

    npz_path = results_dir / f"trial_paths_{run_name}.npz"
    np.savez(npz_path, **{k: np.array(v, dtype=object) for k, v in epoch_to_trajs.items()})

    summary = {
        "run_name": run_name,
        "total_trials": total_trials,
        "accepted_trials": accepted_trials,
        "angle_step": args.angle_step,
        "max_trials": args.max_trials,
        "num_samples": args.num_samples,
        "prep_indices": prep_indices,
        "p_decision_thres": args.p_decision_thres,
        "device": str(device),
        "explained_variance_per_epoch": variance_info,
    }
    summary_path = results_dir / f"trial_paths_{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
