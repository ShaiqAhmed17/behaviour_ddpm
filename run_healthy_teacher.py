#!/usr/bin/env python3
"""
Minimal script to run healthy teacher (no ablation) through analysis.
Generates: results/prospective_memory_dual/index_cued_first_diffusion_0.3_swap_7/timeline_raw_states.npz
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import torch
import colorsys
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart

# Setup paths
REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch

# Constants
PREP_IDX = 2
NEURAL_DIM = 16
N_TRIALS = 200
ANGLE_STEP = 30
SEED = 42
np.random.seed(SEED)

TEACHER_ARGS_PATH = REPO_ROOT / "results_link_sampler/index_cued_first_diffusion_0.3_swap_7/args.yaml"
TEACHER_CKPT_PATH = REPO_ROOT / "results_link_sampler/index_cued_first_diffusion_0.3_swap_7/state.mdl"
RESULTS_DIR = REPO_ROOT / "results/prospective_memory_dual/index_cued_first_diffusion_0.3_swap_7"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate trials
# ─────────────────────────────────────────────────────────────────────────────

def generate_trial_combinations(angle_step=30):
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for color1 in angles:
            for color2 in angles:
                trials.append({
                    'cue': cue,
                    'color1_angle': color1,
                    'color2_angle': color2,
                    'swap': False
                })
    return trials

trials = generate_trial_combinations(ANGLE_STEP)
print(f"Generated {len(trials)} trials")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract neural states helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_neural_state_from_model(trial, task, model, device, prep_idx=2, ablation_vector=None, neural_dim=16):
    with torch.no_grad():
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
        override_stimulus_features_dict = {
            **override_stimulus_features,
            **override_stimulus_cart_features,
            'cued_item_idx': torch.tensor([trial['cue'] - 1]),
        }
        task_variable_dict = task.task_variable_gen.generate_variable_dict(
            batch_size=1, override_stimulus_features_dict=override_stimulus_features_dict
        )
        trial_info = task.generate_trial_information(
            batch_size=1, num_samples=1, override_task_variable_information=task_variable_dict
        )
        prep_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_info.prep_network_inputs]
        diff_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in trial_info.diffusion_network_inputs]
        sample_kwargs = {
            'prep_network_inputs': prep_inputs,
            'diffusion_network_inputs': diff_inputs,
            'prep_epoch_durations': trial_info.prep_epoch_durations,
            'diffusion_epoch_durations': trial_info.diffusion_epoch_durations,
            'samples_shape': [1, 1],
            'noise_scaler': 1.0,
        }
        if ablation_vector is not None:
            sample_kwargs['ablation_vector'] = ablation_vector
        prep_dicts, _ = model.generate_samples(**sample_kwargs)
        return prep_dicts[prep_idx]['postprep_state'][0, 0, :neural_dim].cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load model & task
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nLoading teacher from {TEACHER_ARGS_PATH}")
_, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
    str(TEACHER_ARGS_PATH), device
)
ckpt = torch.load(TEACHER_CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print("✓ Teacher loaded")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Extract states
# ─────────────────────────────────────────────────────────────────────────────

print("\nExtracting neural states...")
neural_states = []
metadata = []

for i, trial in enumerate(trials):
    if i % 100 == 0:
        print(f"  trial {i}/{len(trials)}")
    state = extract_neural_state_from_model(
        trial, task, model, device,
        prep_idx=PREP_IDX,
        ablation_vector=None,
        neural_dim=NEURAL_DIM
    )
    neural_states.append(state)
    metadata.append([trial['cue'], trial['color1_angle'], trial['color2_angle']])

neural_states = np.asarray(neural_states, dtype=np.float32)
metadata = np.asarray(metadata, dtype=np.float32)
print(f"✓ Extracted {len(neural_states)} states with shape {neural_states.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ring geometry analysis (minimal version that matches notebook cell 10)
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning analysis...")

N_BINS = 12
BIN_SIZE = 360.0 / N_BINS

def _bangle(a, sz=30.0):
    return int(a // sz) % int(360 // sz)

def _fit(data):
    ctr = data.mean(0)
    p = PCA(n_components=3).fit(data - ctr)
    n = p.components_[2]
    n /= np.linalg.norm(n)
    return n, ctr, float(p.explained_variance_ratio_[:2].sum()), p.explained_variance_ratio_

def _angle_deg(n1, n2):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), 0, 1))))

# Bin states by cue and target angle
bd = {c: {b: [] for b in range(N_BINS)} for c in [1, 2]}
for i, (cue, c1, c2) in enumerate(metadata):
    cue = int(cue)
    bd[cue][_bangle(c1 if cue == 1 else c2, BIN_SIZE)].append(neural_states[i])

# Average within bins
av = {c: np.full((N_BINS, NEURAL_DIM), np.nan) for c in [1, 2]}
for c in [1, 2]:
    for b in range(N_BINS):
        if bd[c][b]:
            av[c][b] = np.mean(bd[c][b], axis=0)

# Fit planes and extract ring geometry
print("Fitting planes...")
rings_target, rings_distractor = {c: {} for c in [1, 2]}, {c: {} for c in [1, 2]}
for c in [1, 2]:
    # Target: cue=c
    td = av[c]
    if np.any(~np.isnan(td).any(axis=1)):
        tn, tc, *_ = _fit(td[~np.isnan(td).any(axis=1)])
        rings_target[c]['normal'] = tn
        rings_target[c]['center'] = tc
    # Distractor: cue≠c
    dd = av[3 - c]
    if np.any(~np.isnan(dd).any(axis=1)):
        dn, dc, *_ = _fit(dd[~np.isnan(dd).any(axis=1)])
        rings_distractor[c]['normal'] = dn
        rings_distractor[c]['center'] = dc

# Timeline extraction (project each state onto target/distractor planes across timesteps)
print("Extracting timeline states...")
tl_st = []  # shape: (n_trials, T, 16)
tl_mt = []  # shape: (n_trials, 3) = [cue, c1, c2]

for i, trial in enumerate(trials):
    tl_st.append(neural_states[i])
    tl_mt.append([trial['cue'], trial['color1_angle'], trial['color2_angle']])

tl_st = np.asarray(tl_st, dtype=np.float32)
tl_mt = np.asarray(tl_mt, dtype=np.int32)

print(f"  tl_st shape: {tl_st.shape}")
print(f"  tl_mt shape: {tl_mt.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save NPZ
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
npz_path = RESULTS_DIR / 'timeline_raw_states.npz'

np.savez_compressed(
    npz_path,
    tl_st=tl_st,
    tl_mt=tl_mt,
    pep_ref=np.array([PREP_IDX], dtype=np.int32),
    n_diffusion_steps=np.int32(1)
)

print(f"\n✓ Saved NPZ to {npz_path}")
print(f"  Size: {npz_path.stat().st_size / 1e6:.1f} MB")

print("\n✓ Done!")
