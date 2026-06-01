#!/usr/bin/env python3
"""Extract timeline_raw_states.npz for new student models.

Discovers models not yet present in prospective_memory_dual/ and runs
inference on all 288 trials (2 cues × 144 color combinations at 30° steps),
then saves tl_st and tl_mt in the format expected by feature_analysis.py.

New model types handled:
  - multi-dir nullspace:  ...recovery_ablation_{N}-{M}-..._{seed}
  - PCA ablation:         ...recovery_pca_ablation_{dirs}_{seed}
  - no-ablation recovery: ...recovery_no_ablation_{seed}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch
from ddpm.analysis.new_analysis.export_student_sweep_cache import (
    generate_trial_combinations,
    move_to_device,
    build_override_stimulus_features_dict,
)

RESULTS_ROOT = REPO_ROOT / 'results_link_sampler'
OUTPUT_ROOT  = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'
NEURAL_DIM   = 16
ANGLE_STEP   = 30
BATCH_SIZE   = 32

# ─── Pattern matching for new models ─────────────────────────────────────────

_PAT_MULTI  = re.compile(r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(\d+(?:-\d+)+)_(\d+)$')
_PAT_PCA    = re.compile(r'^index_cued_first_diffusion_0\.3_swap_recovery_pca_ablation_(\d+(?:-\d+)*)_(\d+)$')
_PAT_NOABL  = re.compile(r'^index_cued_first_diffusion_0\.3_swap_recovery_no_ablation_(\d+)$')


def _is_new_model(d: Path) -> bool:
    if not d.is_dir():
        return False
    if not (d / 'state.mdl').exists() or not (d / 'args.yaml').exists():
        return False
    return any(p.match(d.name) for p in [_PAT_MULTI, _PAT_PCA, _PAT_NOABL])


def discover_new_models() -> list[Path]:
    new_models = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not _is_new_model(d):
            continue
        out_npz = OUTPUT_ROOT / d.name / 'timeline_raw_states.npz'
        if out_npz.exists():
            print(f'  [skip — already done] {d.name}')
            continue
        new_models.append(d)
    return new_models


def extract_and_save(run_dir: Path, device: torch.device) -> None:
    print(f'\n{"="*70}')
    print(f'Processing: {run_dir.name}')

    args_path = str(run_dir / 'args.yaml')
    ckpt_path = run_dir / 'state.mdl'

    _, task, model, *_ = generate_model_and_task_from_args_path_multiepoch(args_path, device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    trials = generate_trial_combinations(ANGLE_STEP)
    print(f'  Trials: {len(trials)}')

    states_seq_by_prep: Dict[int, List[np.ndarray]] = {}
    metadata_rows: List[List[float]] = []

    with torch.no_grad():
        for start in range(0, len(trials), BATCH_SIZE):
            batch = trials[start:start + BATCH_SIZE]
            bsz   = len(batch)

            override     = build_override_stimulus_features_dict(batch)
            var_dict     = task.task_variable_gen.generate_variable_dict(
                batch_size=bsz, override_stimulus_features_dict=override
            )
            trial_info   = task.generate_trial_information(
                batch_size=bsz, num_samples=1,
                override_task_variable_information=var_dict,
            )

            prep_inputs  = move_to_device(trial_info.prep_network_inputs,      device)
            diff_inputs  = move_to_device(trial_info.diffusion_network_inputs,  device)

            prep_dicts, _ = model.generate_samples(
                prep_network_inputs       = prep_inputs,
                diffusion_network_inputs  = diff_inputs,
                prep_epoch_durations      = trial_info.prep_epoch_durations,
                diffusion_epoch_durations = trial_info.diffusion_epoch_durations,
                samples_shape             = [bsz, 1],
                noise_scaler              = 1.0,
            )

            for prep_idx, pd in enumerate(prep_dicts):
                # shape: (batch, parallel_samples, T_p, D)
                pt = pd['preparatory_trajectory'].detach().cpu().numpy()
                states_seq_by_prep.setdefault(prep_idx, [])
                for i in range(bsz):
                    states_seq_by_prep[prep_idx].append(
                        pt[i, 0, :, :NEURAL_DIM].astype(np.float32)
                    )

            for t in batch:
                metadata_rows.append([int(t['cue']), float(t['color1_angle']), float(t['color2_angle'])])

            if (start // BATCH_SIZE + 1) % 3 == 0:
                print(f'  batch {start // BATCH_SIZE + 1}/{len(trials)//BATCH_SIZE}')

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build tl_st: (n_trials, T_total, NEURAL_DIM)
    prep_indices = sorted(states_seq_by_prep.keys())
    n_trials = len(trials)

    tl_st_list = []
    for i in range(n_trials):
        row = np.concatenate(
            [states_seq_by_prep[pi][i] for pi in prep_indices], axis=0
        )  # (T_total, 16)
        tl_st_list.append(row)

    tl_st = np.stack(tl_st_list, axis=0)   # (n_trials, T_total, 16)
    tl_mt = np.asarray(metadata_rows, dtype=np.float32)  # (n_trials, 3)

    print(f'  tl_st: {tl_st.shape}  tl_mt: {tl_mt.shape}')

    out_dir = OUTPUT_ROOT / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / 'timeline_raw_states.npz'
    np.savez_compressed(out_npz, tl_st=tl_st, tl_mt=tl_mt)
    print(f'  Saved: {out_npz}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    models = discover_new_models()
    if not models:
        print('\nAll new models already have timeline_raw_states.npz — nothing to do.')
        return

    print(f'\nWill process {len(models)} model(s):')
    for m in models:
        print(f'  {m.name}')

    for run_dir in models:
        extract_and_save(run_dir, device)

    print(f'\n{"="*70}')
    print(f'Done.  Outputs in {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
