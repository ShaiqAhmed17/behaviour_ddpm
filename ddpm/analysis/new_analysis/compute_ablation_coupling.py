"""
Compute a principled per-ablation-direction coupling ranking using
precomputed summary JSONs from sampled trajectories.

This script scans `results_link_sampler/` for run folders that contain
`summary_sampled_trajectories.json`, extracts a small set of geometry
metrics (prep separation, plane angle, primary PCA variance) and
computes a standardized coupling score. Outputs CSV at
`ddpm/analysis/new_analysis/results/ablation_coupling_ranking.csv`.

Usage: run from repo root:
    python ddpm/analysis/new_analysis/compute_ablation_coupling.py
"""
import json
import re
from pathlib import Path
import numpy as np
import csv

REPO = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_ROOT = REPO / 'results_link_sampler'
ALT_RESULTS_ROOT = REPO / 'ddpm' / 'analysis' / 'new_analysis' / 'results'
OUT_DIR = REPO / 'ddpm' / 'analysis' / 'new_analysis' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'ablation_coupling_ranking.csv'


def find_summary_files(results_root):
    # look for folders containing summary_sampled_trajectories.json
    summaries = []
    roots = [results_root]
    if ALT_RESULTS_ROOT.exists():
        roots.append(ALT_RESULTS_ROOT)

    for root in roots:
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            candidate = p / 'ablated_teacher_trajectories'
            # some runs keep summary inside results folder
            js1 = p / 'summary_sampled_trajectories.json'
            js2 = candidate / 'summary_sampled_trajectories.json'
            if js1.exists():
                summaries.append(js1)
            elif js2.exists():
                summaries.append(js2)
    return summaries


def parse_run_index(run_name):
    # try to extract a trailing integer (e.g., ..._recovery_8 -> 8)
    m = re.search(r"(\d+)(?!.*\d)", run_name)
    if m:
        return int(m.group(1))
    return None


def load_metrics(json_path):
    with open(json_path, 'r') as f:
        d = json.load(f)
    run_name = d.get('run_name', json_path.parent.name)
    idx = parse_run_index(run_name)
    prep_primary = d.get('prep_idx_primary', None)
    pca_primary = d.get('pca_explained_variance_primary', [None, None, None])[0]
    target_distractor_angle = d.get('target_distractor_plane_angle', None)
    prep_results = d.get('prep_epoch_results', {})
    sep = None
    plane_angle = None
    if prep_primary is not None and str(prep_primary) in prep_results:
        entry = prep_results[str(prep_primary)]
        sep = entry.get('separation', None)
        plane_angle = entry.get('plane_angle', None)

    return {
        'run_name': run_name,
        'index': idx,
        'prep_idx_primary': prep_primary,
        'pca_primary_var': pca_primary,
        'target_distractor_plane_angle': target_distractor_angle,
        'prep_separation': sep,
        'prep_plane_angle': plane_angle,
        'json_path': str(json_path),
    }


def standardize_and_score(rows):
    # build arrays with NaNs handled
    seps = np.array([r['prep_separation'] if r['prep_separation'] is not None else np.nan for r in rows], dtype=float)
    angles = np.array([r['target_distractor_plane_angle'] if r['target_distractor_plane_angle'] is not None else np.nan for r in rows], dtype=float)
    pvars = np.array([r['pca_primary_var'] if r['pca_primary_var'] is not None else np.nan for r in rows], dtype=float)

    # replace NaN with column median to avoid propagation
    def fillnan(x):
        m = np.nanmedian(x)
        x[np.isnan(x)] = m
        return x

    seps = fillnan(seps)
    angles = fillnan(angles)
    pvars = fillnan(pvars)

    # we interpret stronger coupling as larger deviation of prep separation
    sep_dev = np.abs(seps - np.median(seps))

    # standardize each metric
    def z(x):
        xm = np.mean(x)
        xs = np.std(x) if np.std(x) > 0 else 1.0
        return (x - xm) / xs

    z_sep = z(sep_dev)
    z_angle = z(angles)
    z_pv = z(1.0 - pvars)  # smaller primary variance may indicate distortion; invert

    # coupling score: sum of standardized metrics (higher -> stronger coupling)
    scores = z_sep + z_angle + z_pv

    for i, r in enumerate(rows):
        r['sep'] = float(seps[i])
        r['angle'] = float(angles[i])
        r['pca_var'] = float(pvars[i])
        r['z_sep'] = float(z_sep[i])
        r['z_angle'] = float(z_angle[i])
        r['z_pca_var_inv'] = float(z_pv[i])
        r['coupling_score'] = float(scores[i])

    # sort rows descending by coupling_score
    rows_sorted = sorted(rows, key=lambda x: x['coupling_score'], reverse=True)
    return rows_sorted


def write_csv(rows, out_path):
    keys = ['run_name', 'index', 'prep_idx_primary', 'sep', 'angle', 'pca_var', 'z_sep', 'z_angle', 'z_pca_var_inv', 'coupling_score', 'json_path']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in keys}
            w.writerow(row)


def main():
    summaries = find_summary_files(RESULTS_ROOT)
    if not summaries:
        print('No summary_sampled_trajectories.json files found under', RESULTS_ROOT)
        return 1

    rows = []
    for s in summaries:
        try:
            rows.append(load_metrics(s))
        except Exception as e:
            print('Failed to load', s, e)

    scored = standardize_and_score(rows)
    write_csv(scored, OUT_CSV)
    print('Wrote coupling ranking to', OUT_CSV)
    # also print top 6
    print('Top candidates:')
    for r in scored[:6]:
        print(r['run_name'], 'index=', r['index'], 'score=', r['coupling_score'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
