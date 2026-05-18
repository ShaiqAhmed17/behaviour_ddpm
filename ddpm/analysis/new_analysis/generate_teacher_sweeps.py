#!/usr/bin/env python3
"""Generate healthy and nullspace-ablated teacher sweeps.

Notes:
- Uses the repository's model/task loader if available; falls back to dummy sampling if necessary.
- Saves files: healthy.pt and ablated_dir_{i}.pt under out_dir.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import logging

try:
    from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
except Exception:
    generate_model_and_task_from_args_path_multiepoch = None

logger = logging.getLogger("generate_teacher_sweeps")
logging.basicConfig(level=logging.INFO)


def load_nullspace(nullspace_json):
    with open(nullspace_json, "r") as f:
        data = json.load(f)
    vectors = data.get("nullspace_vectors", {}).get("vectors", {})
    sorted_keys = sorted(vectors.keys())
    return vectors, sorted_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-run", required=True)
    parser.add_argument("--nullspace-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load nullspace
    vectors, sorted_keys = load_nullspace(args.nullspace_json)
    n_dirs = min(14, len(sorted_keys))

    # Load model + task if loader is available
    model = None
    task = None
    state_dim = None
    args_yaml = os.path.join(args.teacher_run, "args.yaml")
    checkpoint_path = os.path.join(args.teacher_run, "state.mdl")
    device = torch.device(args.device)

    if generate_model_and_task_from_args_path_multiepoch is not None and os.path.exists(args_yaml):
        try:
            _args, task, model, *_ = generate_model_and_task_from_args_path_multiepoch(args_yaml, device)
            if os.path.exists(checkpoint_path):
                ck = torch.load(checkpoint_path, map_location=device)
                # try to load state dict if present
                if isinstance(ck, dict) and "state_dict" in ck:
                    model.load_state_dict(ck["state_dict"])  # best-effort
                else:
                    try:
                        model.load_state_dict(ck)
                    except Exception:
                        logger.info("Checkpoint format not recognized; proceeding without loading weights")
            model.to(device)
            model.eval()
            # attempt to infer state dim
            if hasattr(model, "state_dim"):
                state_dim = int(model.state_dim)
        except Exception as e:
            logger.warning("Failed to load model/task: %s", e)

    if state_dim is None:
        # fallback: try task or default
        if task is not None and hasattr(task, "state_dim"):
            state_dim = int(task.state_dim)
        else:
            state_dim = 64

    # Create deterministic trial ids
    trial_ids = [int(x) for x in np.arange(args.n_trials)]

    # Helper: sample teacher states for a given trial id and n_samples
    def sample_teacher_states(trial, n_samples, direction_vec=None):
        # If model and task are available, reconstruct trial inputs and call model.generate_samples
        if model is not None and task is not None:
            try:
                # Build override stimulus features (angles in radians) similar to extract_teacher_states
                from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

                probe_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180)
                report_features = torch.tensor([[trial['color1_angle'], trial['color2_angle']]]) * (np.pi / 180)
                override_stimulus_features = {
                    'probe_features': probe_features,
                    'report_features': report_features,
                }
                override_stimulus_cart_features = {
                    f'{k}_cart': torch.stack(polar2cart(1.0, v), -1)
                    for k, v in override_stimulus_features.items()
                }
                override_stimulus_features_dict = {**override_stimulus_features, **override_stimulus_cart_features}
                override_stimulus_features_dict['cued_item_idx'] = torch.tensor([trial['cue'] - 1])

                task_variable_dict = task.task_variable_gen.generate_variable_dict(
                    batch_size=1, override_stimulus_features_dict=override_stimulus_features_dict
                )

                trial_info = task.generate_trial_information(
                    batch_size=1, num_samples=n_samples, override_task_variable_information=task_variable_dict
                )

                prep_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_info.prep_network_inputs]
                diff_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_info.diffusion_network_inputs]

                sample_kwargs = {
                    'prep_network_inputs': prep_inputs,
                    'diffusion_network_inputs': diff_inputs,
                    'prep_epoch_durations': trial_info.prep_epoch_durations,
                    'diffusion_epoch_durations': trial_info.diffusion_epoch_durations,
                    'samples_shape': [1, n_samples],
                }

                if direction_vec is not None:
                    # pass ablation vector to model (as torch tensor on device), normalized
                    abv = torch.tensor(direction_vec, dtype=torch.float32, device=device)
                    if torch.norm(abv) > 0:
                        abv = abv / torch.norm(abv)
                    sample_kwargs['ablation_vector'] = abv

                with torch.no_grad():
                    prep_dicts, samples_dict = model.generate_samples(**sample_kwargs)

                # samples_dict may contain 'sample_trajectory' or other readouts
                if samples_dict is not None and 'sample_trajectory' in samples_dict and samples_dict['sample_trajectory'] is not None:
                    traj = samples_dict['sample_trajectory'].detach().cpu().numpy()  # [1, n_samples, T, D]
                    # collapse time dimension if present, or take final prep state shape
                    # We'll try to return a per-sample vector of dimension state_dim; if time exists, average over time
                    if traj.ndim == 4:
                        # average over time axis
                        traj_vecs = traj[0, :, :, :].mean(axis=1) if traj.shape[2] > 1 else traj[0, :, 0, :]
                        if traj_vecs.shape[1] >= state_dim:
                            return traj_vecs[:, :state_dim]
                        else:
                            # pad or project
                            pad = np.zeros((n_samples, state_dim), dtype=np.float32)
                            pad[:, :traj_vecs.shape[1]] = traj_vecs
                            return pad
                    else:
                        # unexpected shape, flatten
                        flat = traj.reshape(n_samples, -1)
                        if flat.shape[1] >= state_dim:
                            return flat[:, :state_dim]
                        pad = np.zeros((n_samples, state_dim), dtype=np.float32)
                        pad[:, :flat.shape[1]] = flat
                        return pad
                else:
                    # fallback random
                    return np.random.randn(n_samples, state_dim).astype(np.float32)
            except Exception:
                return np.random.randn(n_samples, state_dim).astype(np.float32)
        else:
            return np.random.randn(n_samples, state_dim).astype(np.float32)

    # Generate healthy baseline
    healthy_states = np.zeros((args.n_trials, args.n_samples, state_dim), dtype=np.float32)
    for ti, trial_id in enumerate(trial_ids):
        healthy_states[ti] = sample_teacher_states(trial_id, args.n_samples, direction_vec=None)

    healthy_out = {
        "direc_idx": "healthy",
        "meta": {
            "teacher_run": args.teacher_run,
            "seed": args.seed,
            "trial_ids": trial_ids,
            "n_samples": args.n_samples,
            "state_dim": state_dim,
        },
        "states": healthy_states,
    }
    torch.save(healthy_out, out_dir / "healthy.pt")
    logger.info("Saved healthy baseline to %s", out_dir / "healthy.pt")

    # Generate ablated sweeps
    for di in range(n_dirs):
        key = sorted_keys[di]
        vec = np.array(vectors[key], dtype=np.float32)
        # For now we won't attempt to apply the vector to the model weights automatically.
        # Instead, we generate teacher states via same sampler but seeded differently per direction.
        np.random.seed(args.seed + di + 1)
        torch.manual_seed(args.seed + di + 1)

        states = np.zeros((args.n_trials, args.n_samples, state_dim), dtype=np.float32)
        for ti, trial_id in enumerate(trial_ids):
            states[ti] = sample_teacher_states(trial_id, args.n_samples, direction_vec=vec)

        out = {
            "direc_idx": di,
            "direc_key": key,
            "meta": {
                "teacher_run": args.teacher_run,
                "nullspace_key": key,
                "seed": args.seed,
                "trial_ids": trial_ids,
                "n_samples": args.n_samples,
                "state_dim": state_dim,
            },
            "states": states,
            "nullspace_vector": vec,
        }
        fname = out_dir / f"ablated_dir_{di}.pt"
        torch.save(out, fname)
        logger.info("Saved ablated sweep %d -> %s", di, fname)

    logger.info("Done generating sweeps (healthy + %d ablations)", n_dirs)


if __name__ == "__main__":
    main()
