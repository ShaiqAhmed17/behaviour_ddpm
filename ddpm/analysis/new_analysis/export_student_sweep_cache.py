#!/usr/bin/env python3
"""Export deterministic student sweep trajectories to reusable cache files.

This script mirrors the deterministic cue/color sweep idea used for teacher sweep
artifacts, but saves student forward-pass outputs in a stable cache format
(`.npz` + `.json`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart


def generate_trial_combinations(angle_step: int) -> list[dict[str, float]]:
    """Generate deterministic sweep metadata in cue/color order."""
    if 360 % angle_step != 0:
        raise ValueError(f"angle_step must divide 360 exactly, got {angle_step}")

    angles = list(range(0, 360, angle_step))
    trials: list[dict[str, float]] = []
    for cue in (1, 2):
        for color1 in angles:
            for color2 in angles:
                trials.append(
                    {
                        "cue": cue,
                        "color1_angle": float(color1),
                        "color2_angle": float(color2),
                        "swap": False,
                    }
                )
    return trials


def move_to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def build_override_stimulus_features_dict(batch_trials: list[dict[str, float]]) -> dict[str, torch.Tensor]:
    """Build trial override features for task variable generation."""
    color_arr_deg = np.array(
        [[t["color1_angle"], t["color2_angle"]] for t in batch_trials],
        dtype=np.float32,
    )
    color_arr_rad = torch.tensor(color_arr_deg) * (np.pi / 180.0)

    override_stimulus_features = {
        "probe_features": color_arr_rad,
        "report_features": color_arr_rad,
    }

    override_stimulus_cart_features = {}
    for key, value in override_stimulus_features.items():
        override_stimulus_cart_features[f"{key}_cart"] = torch.stack(polar2cart(1.0, value), -1)

    features_dict: dict[str, torch.Tensor] = {}
    features_dict.update(override_stimulus_features)
    features_dict.update(override_stimulus_cart_features)

    # Convert cue {1,2} -> cued_item_idx {0,1}
    features_dict["cued_item_idx"] = torch.tensor(
        [int(t["cue"]) - 1 for t in batch_trials],
        dtype=torch.long,
    )
    return features_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic student sweep cache")
    parser.add_argument("--args-path", type=Path, required=True, help="Path to args.yaml")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to state/best checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for cache files",
    )
    parser.add_argument("--angle-step", type=int, default=30, help="Color angle step in degrees")
    parser.add_argument("--num-samples", type=int, default=32, help="Samples per trial")
    parser.add_argument("--batch-size", type=int, default=32, help="Trials per forward pass")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--save-pt-shards",
        action="store_true",
        help="Also save per-batch .pt shards similar to teacher sweep naming",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache files if present",
    )
    return parser.parse_args()


def _sanitize_epoch_durations(durations: Any) -> tuple[np.ndarray, list[int | None]]:
    """Convert epoch durations into int array with -1 sentinel for None.

    Some tasks can emit values like [None] for diffusion_epoch_durations.
    This helper preserves raw values for JSON while producing a stable numeric
    array for NPZ storage.
    """
    if durations is None:
        raw: list[int | None] = []
    elif isinstance(durations, (list, tuple)):
        raw = [None if d is None else int(d) for d in durations]
    else:
        raw = [int(durations)]

    sanitized = np.asarray([-1 if d is None else int(d) for d in raw], dtype=np.int32)
    return sanitized, raw


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.args_path.exists():
        raise FileNotFoundError(f"args path not found: {args.args_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {args.checkpoint_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "student_sweep_cache.npz"
    summary_path = output_dir / "student_sweep_cache_summary.json"
    shards_dir = output_dir / "student_sweep_pt_shards"

    if not args.force:
        existing = [p for p in (npz_path, summary_path) if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output file(s) already exist: {names}. Use --force to overwrite."
            )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model/task from: {args.args_path}")
    print(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Using device: {device}")

    _, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(str(args.args_path), device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    trials = generate_trial_combinations(args.angle_step)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    print(f"Deterministic trials: {len(trials)}")
    print(f"Batch size: {args.batch_size}, num_samples: {args.num_samples}")

    trajectory_chunks: list[torch.Tensor] = []
    prep_post_chunks: list[torch.Tensor] = []

    prep_epoch_durations_ref = None
    diffusion_epoch_durations_ref = None

    if args.save_pt_shards:
        shards_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(trials), args.batch_size):
        batch_trials = trials[start : start + args.batch_size]
        batch_size = len(batch_trials)
        sweep_step = start // args.batch_size

        override_features = build_override_stimulus_features_dict(batch_trials)
        task_variable_dict = task.task_variable_gen.generate_variable_dict(
            batch_size=batch_size,
            override_stimulus_features_dict=override_features,
        )

        trial_info = task.generate_trial_information(
            batch_size=batch_size,
            num_samples=args.num_samples,
            override_task_variable_information=task_variable_dict,
        )

        prep_inputs_device = move_to_device(trial_info.prep_network_inputs, device)
        diffusion_inputs_device = move_to_device(trial_info.diffusion_network_inputs, device)

        with torch.no_grad():
            prep_dicts, samples_dict = model.generate_samples(
                prep_network_inputs=prep_inputs_device,
                diffusion_network_inputs=diffusion_inputs_device,
                prep_epoch_durations=trial_info.prep_epoch_durations,
                diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
                samples_shape=[batch_size, args.num_samples],
                noise_scaler=1.0,
            )

        traj_key = (
            "embedded_sample_trajectory"
            if "embedded_sample_trajectory" in samples_dict
            else "sample_trajectory"
        )
        if traj_key not in samples_dict or samples_dict[traj_key] is None:
            raise RuntimeError("No trajectory key found in samples_dict")

        trajectories = samples_dict[traj_key][..., :16].detach().cpu().float()
        prep_post = torch.stack(
            [prep_dict["postprep_state"][..., :16].detach().cpu().float() for prep_dict in prep_dicts],
            dim=2,
        )

        trajectory_chunks.append(trajectories)
        prep_post_chunks.append(prep_post)

        if prep_epoch_durations_ref is None:
            prep_epoch_durations_ref, prep_epoch_durations_raw = _sanitize_epoch_durations(
                trial_info.prep_epoch_durations
            )
            diffusion_epoch_durations_ref, diffusion_epoch_durations_raw = _sanitize_epoch_durations(
                trial_info.diffusion_epoch_durations
            )

        if args.save_pt_shards:
            shard_payload = {
                "student_trajectories": trajectories,
                "student_prep_post_states": prep_post,
                "sweep_batch_trials": batch_trials,
                "sweep_step": int(sweep_step),
            }
            shard_path = shards_dir / f"student_trajectories_sweep_step_{sweep_step:06d}.pt"
            torch.save(shard_payload, shard_path)

        print(
            f"Processed batch {sweep_step:04d}: "
            f"trials {start}-{start + batch_size - 1}, "
            f"traj shape {tuple(trajectories.shape)}"
        )

    all_trajectories = torch.cat(trajectory_chunks, dim=0).numpy()
    all_prep_post = torch.cat(prep_post_chunks, dim=0).numpy()

    metadata = np.array(
        [[int(t["cue"]), float(t["color1_angle"]), float(t["color2_angle"])] for t in trials],
        dtype=np.float32,
    )

    np.savez_compressed(
        npz_path,
        trajectories=all_trajectories,
        prep_post_states=all_prep_post,
        metadata=metadata,
        prep_epoch_durations=prep_epoch_durations_ref,
        diffusion_epoch_durations=diffusion_epoch_durations_ref,
        angle_step=np.int32(args.angle_step),
        num_samples=np.int32(args.num_samples),
        neural_dim=np.int32(16),
    )

    summary = {
        "args_path": str(args.args_path),
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(output_dir),
        "npz_path": str(npz_path),
        "saved_pt_shards": bool(args.save_pt_shards),
        "pt_shards_dir": str(shards_dir) if args.save_pt_shards else None,
        "n_trials": int(metadata.shape[0]),
        "num_samples": int(args.num_samples),
        "angle_step": int(args.angle_step),
        "trajectories_shape": list(all_trajectories.shape),
        "prep_post_states_shape": list(all_prep_post.shape),
        "metadata_shape": list(metadata.shape),
        "prep_epoch_durations": prep_epoch_durations_ref.tolist() if prep_epoch_durations_ref is not None else [],
        "diffusion_epoch_durations": diffusion_epoch_durations_ref.tolist() if diffusion_epoch_durations_ref is not None else [],
        "prep_epoch_durations_raw": prep_epoch_durations_raw if prep_epoch_durations_ref is not None else [],
        "diffusion_epoch_durations_raw": diffusion_epoch_durations_raw if diffusion_epoch_durations_ref is not None else [],
        "device": str(device),
        "seed": int(args.seed),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nExport complete")
    print(f"- Cache file:   {npz_path}")
    print(f"- Summary file: {summary_path}")
    if args.save_pt_shards:
        print(f"- PT shards:    {shards_dir}")


if __name__ == "__main__":
    main()
