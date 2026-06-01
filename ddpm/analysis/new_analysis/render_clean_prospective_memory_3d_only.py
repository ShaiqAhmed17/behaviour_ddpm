#!/usr/bin/env python3
"""Render a clean prospective-memory 3D-only figure with one global legend.

This script regenerates the healthy-student prospective panel layout directly
from the model for the default student recovery run used by the notebook.
It keeps the pre-stimulus panels neutral gray and color-codes the post-stimulus
panels by cued-color bin.
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from ddpm.utils.vis.style import set_publication_style
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

set_publication_style()


ARGS_PATH = REPO_ROOT / "results_link_sampler/index_cued_first_diffusion_0.3_swap_7/args.yaml"
CKPT_PATH = REPO_ROOT / "results_link_sampler/index_cued_first_diffusion_0.3_swap_7/state.mdl"
OUT_PATH = (
    REPO_ROOT
    / "ddpm/analysis/new_analysis/results/prospective_memory_dual/healthy_student/"
    / "prospective_memory_all_prep_indices_3d_only_clean.png"
)

N_BINS = 12
PREP_INDICES = [0, 1, 2, 3]
ANGLES = list(range(0, 360, 30))


def angle_to_colour(bin_idx: int, n_bins: int = 12):
    hue = bin_idx / n_bins
    return colorsys.hsv_to_rgb(hue, 0.9, 0.9)


def bin_angle(angle: float, bin_size: float = 30.0) -> int:
    return int(angle // bin_size) % int(360 // bin_size)


def bin_and_average_by_cued_color(states: np.ndarray, metadata: np.ndarray, n_bins: int = 12):
    bin_size = 360.0 / n_bins
    binned = {1: {b: [] for b in range(n_bins)}, 2: {b: [] for b in range(n_bins)}}

    for i, (cue, c1, c2) in enumerate(metadata):
        cue = int(cue)
        cued_angle = c1 if cue == 1 else c2
        b = bin_angle(cued_angle, bin_size)
        binned[cue][b].append(states[i])

    averaged = {
        1: np.full((n_bins, states.shape[1]), np.nan, dtype=np.float32),
        2: np.full((n_bins, states.shape[1]), np.nan, dtype=np.float32),
    }

    for cue in [1, 2]:
        for b in range(n_bins):
            if binned[cue][b]:
                averaged[cue][b] = np.mean(binned[cue][b], axis=0)

    return averaged


def extract_neural_state_from_model(trial, task, model, device, prep_idx=2, neural_dim=16):
    with torch.no_grad():
        probe_features = torch.tensor([[trial["color1_angle"], trial["color2_angle"]]], dtype=torch.float32)
        report_features = torch.tensor([[trial["color1_angle"], trial["color2_angle"]]], dtype=torch.float32)
        probe_features = probe_features * (np.pi / 180)
        report_features = report_features * (np.pi / 180)

        override_stimulus_features = {
            "probe_features": probe_features,
            "report_features": report_features,
        }
        override_stimulus_cart_features = {
            f"{k}_cart": torch.stack(polar2cart(1.0, v), -1)
            for k, v in override_stimulus_features.items()
        }
        override_stimulus_features_dict = {
            **override_stimulus_features,
            **override_stimulus_cart_features,
            "cued_item_idx": torch.tensor([trial["cue"] - 1]),
        }

        task_variable_dict = task.task_variable_gen.generate_variable_dict(
            batch_size=1,
            override_stimulus_features_dict=override_stimulus_features_dict,
        )
        trial_info = task.generate_trial_information(
            batch_size=1,
            num_samples=1,
            override_task_variable_information=task_variable_dict,
        )

        prep_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_info.prep_network_inputs]
        diff_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_info.diffusion_network_inputs]

        prep_dicts, _ = model.generate_samples(
            prep_network_inputs=prep_inputs,
            diffusion_network_inputs=diff_inputs,
            prep_epoch_durations=trial_info.prep_epoch_durations,
            diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
            samples_shape=[1, 1],
            noise_scaler=1.0,
        )

        return prep_dicts[prep_idx]["postprep_state"][0, 0, :neural_dim].cpu().numpy().astype(np.float32)


def build_trials():
    trials = []
    for cue in [1, 2]:
        for color1 in ANGLES:
            for color2 in ANGLES:
                trials.append({"cue": cue, "color1_angle": color1, "color2_angle": color2})
    return trials


def main():
    if not ARGS_PATH.exists():
        raise FileNotFoundError(f"Missing args file: {ARGS_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {CKPT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(str(ARGS_PATH), device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    trials = build_trials()

    # Phase 1: extract and bin for all prep indices, store averaged results
    averaged_per_prep = {}
    for prep_idx in PREP_INDICES:
        states = []
        metadata = []
        for trial in trials:
            states.append(extract_neural_state_from_model(trial, task, model, device, prep_idx=prep_idx))
            metadata.append([trial["cue"], trial["color1_angle"], trial["color2_angle"]])

        states = np.asarray(states, dtype=np.float32)
        metadata = np.asarray(metadata, dtype=np.float32)

        averaged = bin_and_average_by_cued_color(states, metadata, n_bins=N_BINS)
        averaged_per_prep[prep_idx] = averaged

    # Build a global dataset (only valid bins) for fitting a single PCA
    global_rows = []
    for prep_idx in PREP_INDICES:
        averaged = averaged_per_prep[prep_idx]
        for cue in [1, 2]:
            arr = np.asarray(averaged[cue])
            valid = arr[~np.isnan(arr).any(axis=1)]
            if valid.size:
                global_rows.append(valid)
    if not global_rows:
        raise RuntimeError("No valid binned data found to fit global PCA")
    global_all_binned = np.vstack(global_rows)

    pca = PCA(n_components=3)
    pca.fit(global_all_binned)
    var_ratio = pca.explained_variance_ratio_

    # Phase 2: plotting using the shared PCA
    fig = plt.figure(figsize=(10, 18))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.10, hspace=0.28)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=8, label="cue 1"),
        Line2D([0], [0], marker="^", linestyle="None", color="w", markerfacecolor="lightgray", markeredgecolor="k", markersize=8, label="cue 2"),
    ]

    colors = np.array([angle_to_colour(b, N_BINS) for b in range(N_BINS)])

    for row_idx, prep_idx in enumerate(PREP_INDICES):
        averaged = averaged_per_prep[prep_idx]

        # Collect valid points per cue and their bin indices
        cue_coords = {1: [], 2: []}
        cue_bins = {1: [], 2: []}
        for cue in [1, 2]:
            for b in range(N_BINS):
                vec = averaged[cue][b]
                if not np.isnan(vec).any():
                    cue_coords[cue].append(vec)
                    cue_bins[cue].append(b)

        ax = fig.add_subplot(4, 1, row_idx + 1, projection="3d")

        for cue, marker in [(1, "o"), (2, "^")]:
            coords_in = np.asarray(cue_coords[cue])
            if coords_in.size == 0:
                continue
            coords_pca = pca.transform(coords_in)
            if prep_idx < 2:
                plot_colors = np.tile(np.array([[0.60, 0.60, 0.60]]), (coords_pca.shape[0], 1))
            else:
                plot_colors = np.array([angle_to_colour(b, N_BINS) for b in cue_bins[cue]])
            ax.scatter(coords_pca[:, 0], coords_pca[:, 1], coords_pca[:, 2], c=plot_colors, marker=marker, s=80, edgecolors="k", linewidths=1.0, alpha=0.9)

        ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=10)
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=10)
        ax.set_zlabel(f"PC3 ({var_ratio[2]:.1%})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=24, azim=-56)

    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()