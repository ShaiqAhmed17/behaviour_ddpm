#!/usr/bin/env python3
"""Regenerate prep-trajectory 3D PCA plots from timeline_raw_states.npz files."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.analysis.new_analysis.prospective_memory_plotting import (
    make_full_step_trajectory_dataset,
    plot_global_3d_trajectories,
)
from ddpm.utils.vis.style import set_publication_style

set_publication_style()


RESULTS = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'
N_BINS = 12


def replot_dir(d):
    npz_path = d / 'timeline_raw_states.npz'
    data = np.load(npz_path)
    tl_st = data['tl_st']   # (n_trials, T, D)
    tl_mt = data['tl_mt']   # (n_trials, 3)
    if 'pep_ref' in data:
        pep_ref = data['pep_ref']
        n_diff = int(data.get('n_diffusion_steps', 0))
        tl_st = tl_st[:, :int(pep_ref.sum()), :]  # use only prep portion
    else:
        # Older format: only prep timesteps saved; use known default epoch split
        pep_ref = np.array([5, 15, 5, 1], dtype=np.int32) if tl_st.shape[1] == 26 else np.array([tl_st.shape[1]], dtype=np.int32)

    states_seq_by_prep = {}
    start = 0
    for epoch_idx, epoch_len in enumerate(pep_ref):
        epoch_len = int(epoch_len)
        states_seq_by_prep[epoch_idx] = [tl_st[i, start:start + epoch_len, :] for i in range(len(tl_st))]
        start += epoch_len

    prep_points, prep_labels, _ = make_full_step_trajectory_dataset(
        states_seq_by_prep=states_seq_by_prep,
        metadata=tl_mt,
        prep_indices=list(range(len(pep_ref))),
        n_bins=N_BINS,
    )

    if prep_points.shape[0] < 3:
        print(f'  SKIP (too few points): {d.name}')
        return

    n_comp = min(3, prep_points.shape[0], prep_points.shape[1])
    pca = PCA(n_components=n_comp)
    prep_coords = pca.fit_transform(prep_points)

    out_path = d / 'prospective_memory_all_prep_indices'
    plot_global_3d_trajectories(
        pca_coords=prep_coords,
        labels=prep_labels,
        pca=pca,
        n_bins=N_BINS,
        out_path=str(out_path),
        title='Preparatory trajectories',
        label_schema='prep',
    )


if __name__ == '__main__':
    dirs = sorted(
        d for d in RESULTS.iterdir()
        if d.is_dir() and (d / 'timeline_raw_states.npz').exists()
    )
    print(f'Found {len(dirs)} directories with timeline_raw_states.npz')
    for i, d in enumerate(dirs, 1):
        print(f'[{i}/{len(dirs)}] {d.name}')
        replot_dir(d)
    print('Done.')
