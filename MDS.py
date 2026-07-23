#!/usr/bin/env python3
"""
MDS.py  —  unified MDS over three distance types.

Usage:
    python MDS.py --kind sw           # Sliced-Wasserstein (metric)
    python MDS.py --kind procrustes   # Procrustes residual (non-metric)
    python MDS.py --kind feature      # Feature-space Euclidean (non-metric)
    python MDS.py --kind all          # All three
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.vis.style import set_publication_style, save_figure, save_legend

set_publication_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS    = REPO_ROOT / 'ddpm/analysis/new_analysis/results'
FEAT_DIR   = RESULTS / 'prospective_memory_dual'
OUTPUT_DIR = RESULTS / 'mds_unified'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Canonical model list ──────────────────────────────────────────────────────
# Each entry: (label, role, abl_dir, student_type)
#   label        – internal key used throughout
#   role         – 'teacher' | 'student'
#   abl_dir      – int ablation direction, or None for healthy
#   student_type – 'single' | 'none' | None (teachers)
MODELS = [
    ('T_healthy', 'teacher', None,  None),
    ('T_d01',     'teacher', 1,     None),
    ('T_d04',     'teacher', 4,     None),
    ('T_d05',     'teacher', 5,     None),
    ('T_d07',     'teacher', 7,     None),
    ('T_d12',     'teacher', 12,    None),
    ('S_d01_s0',  'student', 1,     'single'),
    ('S_d01_s1',  'student', 1,     'single'),
    ('S_d01_s2',  'student', 1,     'single'),
    ('S_d01_s3',  'student', 1,     'single'),
    ('S_d01_s4',  'student', 1,     'single'),
    ('S_d04_s0',  'student', 4,     'single'),
    ('S_d05_s0',  'student', 5,     'single'),
    ('S_d07_s0',  'student', 7,     'single'),
    ('S_d12_s0',  'student', 12,    'single'),
    ('S_d12_s1',  'student', 12,    'single'),
    ('S_noabl',   'student', None,  'none'),
    ('S_noabl_1', 'student', None,  'none'),
]

N  = len(MODELS)
_idx = {m[0]: i for i, m in enumerate(MODELS)}

# ── Label maps to each data source ────────────────────────────────────────────
# SW  (pairwise_wasserstein.npy)
_SW = {
    'T_healthy': 'teacher_healthy',
    'T_d01':     'teacher_dir01',
    'T_d04':     'teacher_dir04',
    'T_d05':     'teacher_dir05',
    'T_d07':     'teacher_dir07',
    'T_d12':     'teacher_dir12',
    'S_d01_s0':  'student_dir01_s0',
    'S_d01_s1':  'student_dir01_s1',
    'S_d01_s2':  'student_dir01_s2',
    'S_d01_s3':  'student_dir01_s3',
    'S_d01_s4':  'student_dir01_s4',
    'S_d04_s0':  'student_dir04_s0',
    'S_d05_s0':  'student_dir05_s0',
    'S_d07_s0':  'student_dir07_s0',
    'S_d12_s0':  'student_dir12_s0',
    'S_d12_s1':  'student_dir12_s1',
    'S_d13_s0':  'student_dir13_s0',
    'S_noabl':   'student_noabl_s0',
    'S_noabl_1': 'student_noabl_abl_s0',
}

# Procrustes — teacher-teacher block
_PROC_TT = {
    'T_healthy': 'unablated',
    'T_d01':     'dir_01',
    'T_d04':     'dir_04',
    'T_d05':     'dir_05',
    'T_d07':     'dir_07',
    'T_d12':     'dir_12',
}
# Procrustes — student-student block (normalized: abl{d}_s{s} → abl{d}_{s})
_PROC_SS = {
    'S_d01_s0': 'abl1_0',  'S_d01_s1': 'abl1_1',
    'S_d01_s2': 'abl1_2',  'S_d01_s3': 'abl1_3',  'S_d01_s4': 'abl1_4',
    'S_d04_s0': 'abl4_0',  'S_d05_s0': 'abl5_0',  'S_d07_s0': 'abl7_0',
    'S_d12_s0': 'abl12_0', 'S_d12_s1': 'abl12_1',
    'S_noabl':  'noabl_s0',
    'S_noabl_1': 'noabl_abl_s0',
}
# Procrustes — teacher-student block (teacher labels same as TT, student same as SS)
_PROC_TS_T = _PROC_TT
_PROC_TS_S = _PROC_SS

# Feature — path to timeline_raw_states.npz
def _feat_npz(label, abl_dir, role, student_type):
    base = FEAT_DIR
    if role == 'teacher':
        if abl_dir is None:
            return base / 'index_cued_first_diffusion_0.3_swap_7/timeline_raw_states.npz'
        return base / f'ablated_teacher_dir_{abl_dir:02d}/timeline_raw_states.npz'
    # student
    seed = int(re.search(r's(\d+)$', label).group(1)) if re.search(r's(\d+)$', label) else 0
    if student_type == 'none':
        if label == 'S_noabl_1':
            return base / 'index_cued_first_diffusion_0.3_swap_recovery_ablation_no_ablation_0/timeline_raw_states.npz'
        return base / 'index_cued_first_diffusion_0.3_swap_recovery_no_ablation_0/timeline_raw_states.npz'
    return base / f'index_cued_first_diffusion_0.3_swap_recovery_ablation_{abl_dir}_{seed}/timeline_raw_states.npz'

_FEAT_NPZ = {
    lab: _feat_npz(lab, abl, role, stype)
    for lab, role, abl, stype in MODELS
}

# ── Distance matrix loaders ───────────────────────────────────────────────────

def load_sw() -> np.ndarray:
    """Sliced-Wasserstein pairwise distances (from cached .npy)."""
    all_labels = np.load(RESULTS / 'wasserstein_mds/model_labels.npy', allow_pickle=True).tolist()
    all_mat    = np.load(RESULTS / 'wasserstein_mds/pairwise_wasserstein.npy')
    idx = {l: i for i, l in enumerate(all_labels)}
    D = np.zeros((N, N))
    for lab, *_ in MODELS:
        for lab2, *_ in MODELS:
            D[_idx[lab], _idx[lab2]] = all_mat[idx[_SW[lab]], idx[_SW[lab2]]]
    return D


def load_procrustes() -> np.ndarray:
    """Procrustes residuals assembled from TT, SS, and TS blocks."""
    tt_mat = np.load(RESULTS / 'teacher_teacher_heatmap/heatmap_fixed.npy')
    tt_idx = {str(l): i for i, l in enumerate(
        np.load(RESULTS/'teacher_teacher_heatmap/labels.npz', allow_pickle=True)['labels'])}

    ss_raw   = np.load(RESULTS / 'student_student_heatmap/labels.npz', allow_pickle=True)['labels']
    def _norm(l):
        l = str(l)
        if l == 'no_abl': return 'noabl_s0'
        m = re.match(r'abl(\d+)_s(\d+)', l)
        return f'abl{m.group(1)}_{m.group(2)}' if m else l
    ss_mat = np.load(RESULTS / 'student_student_heatmap/heatmap_fixed.npy')
    ss_idx  = {_norm(l): i for i, l in enumerate(ss_raw)}

    ts_raw  = np.load(RESULTS / 'procrustes_heatmap_15dirs_all_students/labels.npz', allow_pickle=True)
    ts_t_idx = {str(l): i for i, l in enumerate(ts_raw['teacher_labels'])}
    ts_s_idx = {str(l): i for i, l in enumerate(ts_raw['student_labels'])}
    ts_mat   = np.load(RESULTS / 'procrustes_heatmap_15dirs_all_students/heatmap_fixed.npy')

    D = np.zeros((N, N))
    teachers = [(lab, i) for i, (lab, role, *_) in enumerate(MODELS) if role == 'teacher']
    students = [(lab, i) for i, (lab, role, *_) in enumerate(MODELS) if role == 'student']

    for la, ia in teachers:
        for lb, ib in teachers:
            D[ia, ib] = tt_mat[tt_idx[_PROC_TT[la]], tt_idx[_PROC_TT[lb]]]
    for la, ia in students:
        for lb, ib in students:
            ka, kb = _PROC_SS[la], _PROC_SS[lb]
            if ka not in ss_idx or kb not in ss_idx:
                D[ia, ib] = np.nan
            else:
                D[ia, ib] = ss_mat[ss_idx[ka], ss_idx[kb]]
    for lt, it in teachers:
        for ls, is_ in students:
            ks = _PROC_TS_S[ls]
            if ks not in ts_s_idx:
                D[it, is_] = np.nan
                D[is_, it] = np.nan
            else:
                v = ts_mat[ts_t_idx[_PROC_TS_T[lt]], ts_s_idx[ks]]
                D[it, is_] = v
                D[is_, it] = v

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _ring_metrics_over_time(tgt_bm, dis_bm):
    from sklearn.decomposition import PCA as _PCA
    N_BINS = 12
    T = tgt_bm.shape[1]
    mr  = np.full((2, T), np.nan, np.float32)
    rs  = np.full((2, T), np.nan, np.float32)
    ecc = np.full((2, T), np.nan, np.float32)
    arc = np.full((2, T), np.nan, np.float32)
    plan= np.full((2, T), np.nan, np.float32)
    cnrm= np.full((2, T), np.nan, np.float32)
    vpc = np.full((2, T), np.nan, np.float32)
    pang= np.full(T, np.nan, np.float32)
    csep= np.full(T, np.nan, np.float32)
    vcom= np.full(T, np.nan, np.float32)
    for t in range(T):
        tp, dp = tgt_bm[:, t, :], dis_bm[:, t, :]
        comb   = np.vstack([tp, dp])
        cpca   = _PCA(n_components=3)
        coords = cpca.fit_transform(comb)
        t3, d3 = coords[:N_BINS], coords[N_BINS:]
        vcom[t] = float(cpca.explained_variance_ratio_[:2].sum())
        tc, dc = t3.mean(0), d3.mean(0)
        td = np.linalg.norm(t3 - tc, axis=1)
        dd = np.linalg.norm(d3 - dc, axis=1)
        mr[0,t],mr[1,t]   = td.mean(), dd.mean()
        rs[0,t],rs[1,t]   = td.std(),  dd.std()
        cnrm[0,t]         = np.linalg.norm(tc)
        cnrm[1,t]         = np.linalg.norm(dc)
        csep[t]           = np.linalg.norm(tc - dc)
        nt = nd = None
        for ri, (pts, ctr) in enumerate([(t3, tc), (d3, dc)]):
            p = _PCA(n_components=3).fit(pts - ctr)
            plan[ri,t] = float(p.explained_variance_ratio_[:2].sum())
            vpc[ri,t]  = float(_PCA(n_components=2).fit(pts).explained_variance_ratio_.sum())
            if ri == 0: nt = p.components_[2]
            else:       nd = p.components_[2]
        pang[t] = float(np.degrees(np.arccos(np.clip(abs(np.dot(nt, nd)), 0, 1))))
        def _ecc(pts):
            ev = _PCA(n_components=2).fit(pts).explained_variance_
            return ev[0]/ev[1] if ev[1] > 1e-12 else np.nan
        ecc[0,t],ecc[1,t] = _ecc(t3), _ecc(d3)
        def _arc(pts):
            return float(np.std([np.linalg.norm(pts[(i+1)%N_BINS]-pts[i]) for i in range(N_BINS)]))
        arc[0,t],arc[1,t] = _arc(t3), _arc(d3)
    return dict(mean_radius=mr, radius_std=rs, eccentricity=ecc, arc_std=arc,
                planarity=plan, centroid_norm=cnrm, var_pc12=vpc,
                plane_angle_deg=pang, centroid_separation=csep, var_pc12_combined=vcom)


def extract_all_features(npz_path: Path) -> dict:
    from sklearn.decomposition import PCA as _PCA
    N_BINS, BIN_SZ = 12, 30.0
    npz  = np.load(npz_path)
    tl_st, tl_mt = npz['tl_st'], npz['tl_mt']

    def _bangle(a): return int(a // BIN_SZ) % N_BINS
    def _bin_avg(states_t, idx_per_bin):
        T, D = states_t.shape[1], states_t.shape[2]
        out = np.zeros((N_BINS, T, D), np.float32)
        for b, idx in idx_per_bin.items():
            if len(idx): out[b] = states_t[idx].mean(0)
        return out

    g = {c: {'target': {b: [] for b in range(N_BINS)},
              'distractor': {b: [] for b in range(N_BINS)}} for c in [1,2]}
    for i, (cue, c1, c2) in enumerate(tl_mt):
        cue = int(cue)
        ta, da = (c1, c2) if cue == 1 else (c2, c1)
        g[cue]['target'][_bangle(ta)].append(i)
        g[cue]['distractor'][_bangle(da)].append(i)
    for c in [1,2]:
        for role in ['target','distractor']:
            for b in range(N_BINS):
                g[c][role][b] = np.array(g[c][role][b], dtype=np.int64)

    metrics_per_cue = {}
    for c in [1, 2]:
        tgt_bm = _bin_avg(tl_st, g[c]['target'])
        dis_bm = _bin_avg(tl_st, g[c]['distractor'])
        metrics_per_cue[c] = _ring_metrics_over_time(tgt_bm, dis_bm)

    # Topological features
    X = tl_st.reshape(-1, tl_st.shape[-1]).astype(np.float32)
    pca = _PCA(); pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    topo = {
        'intrinsic_dim_90pct': float(np.argmax(cumsum >= 0.9) + 1) / len(cumsum),
        'participation_ratio': float((pca.explained_variance_.sum()**2) / (pca.explained_variance_**2).sum()),
        'var_top3_pcs': float(pca.explained_variance_ratio_[:3].sum()),
    }

    features = {}
    metric_names = ['mean_radius','radius_std','eccentricity','arc_std','planarity',
                    'centroid_norm','var_pc12','plane_angle_deg','centroid_separation','var_pc12_combined']
    for m in metric_names:
        v1, v2 = metrics_per_cue[1][m].mean(), metrics_per_cue[2][m].mean()
        features[f'{m}_diff'] = float(v1 - v2)
        features[f'{m}_sum']  = float(v1 + v2)
    for k, v in topo.items():
        features[f'{k}_sum']  = v
        features[f'{k}_diff'] = 0.0
    return features


def _load_raw_features():
    """Return (fv_matrix [N_models x N_feats], feat_names, model_labels)."""
    feat_cache = OUTPUT_DIR / 'feature_cache.npz'
    if feat_cache.exists():
        cache     = np.load(feat_cache, allow_pickle=True)
        fv_matrix = cache['matrix']
        feat_names = cache['feat_names'].tolist()
        print(f'  Loaded raw features from cache: {fv_matrix.shape}')
    else:
        print('  Extracting features from NPZ files...')
        fvs = []
        for lab, *_ in MODELS:
            p = _FEAT_NPZ[lab]
            print(f'    {lab}...', end=' ', flush=True)
            fvs.append(extract_all_features(p))
            print('OK')
        feat_names = list(fvs[0].keys())
        fv_matrix  = np.array([[f[k] for k in feat_names] for f in fvs], dtype=np.float32)
        np.savez(feat_cache, matrix=fv_matrix, feat_names=np.array(feat_names))
    model_labels = [m[0] for m in MODELS]
    return fv_matrix, feat_names, model_labels


def load_feature(top_k: int = 15) -> np.ndarray:
    """Feature-space Euclidean distances using top-k CV-selected features."""
    fv_matrix, feat_names, _ = _load_raw_features()

    teacher_rows = np.array([i for i, (_, role, *__) in enumerate(MODELS)
                             if role == 'teacher'])
    teacher_fv   = fv_matrix[teacher_rows]
    eps          = 1e-8
    feat_cv      = teacher_fv.std(axis=0) / (np.abs(teacher_fv.mean(axis=0)) + eps)
    top_k_idx    = np.argsort(feat_cv)[-top_k:]

    fv_selected = fv_matrix[:, top_k_idx]
    fv_scaled   = StandardScaler().fit_transform(fv_selected)
    D = squareform(pdist(fv_scaled, metric='euclidean'))
    print(f'  Feature MDS: top-{top_k} CV-selected features, D shape {D.shape}')
    return D


def save_feature_stats(top_k: int = 15):
    """Save a CSV ranking all features by CV (over teachers), with mean/std/min/max."""
    fv_matrix, feat_names, _ = _load_raw_features()

    teacher_rows = np.array([i for i, (_, role, *__) in enumerate(MODELS)
                             if role == 'teacher'])
    teacher_fv = fv_matrix[teacher_rows]
    eps        = 1e-8
    mean_  = teacher_fv.mean(axis=0)
    std_   = teacher_fv.std(axis=0)
    cv_    = std_ / (np.abs(mean_) + eps)
    min_   = teacher_fv.min(axis=0)
    max_   = teacher_fv.max(axis=0)

    rank = np.argsort(cv_)[::-1]  # highest CV first
    df = pd.DataFrame({
        'rank':     np.arange(1, len(feat_names) + 1),
        'feature':  [feat_names[i] for i in rank],
        'cv':       cv_[rank],
        'mean':     mean_[rank],
        'std':      std_[rank],
        'min':      min_[rank],
        'max':      max_[rank],
        'selected': ['yes' if r < top_k else 'no' for r in range(len(feat_names))],
    })

    out = OUTPUT_DIR / 'feature_stats.csv'
    df.to_csv(out, index=False, float_format='%.6g')
    print(f'  Saved feature stats: {out}')
    print(df.to_string(index=False))

# ── MDS fitting ───────────────────────────────────────────────────────────────

def fit_mds(D: np.ndarray, metric: bool) -> np.ndarray:
    mds = MDS(n_components=2, dissimilarity='precomputed', metric=metric, random_state=42)
    coords = mds.fit_transform(D)
    kind = 'metric' if metric else 'non-metric'
    if metric:
        kruskal = np.sqrt(mds.stress_ / (D ** 2).sum())
    else:
        kruskal = mds.stress_
    print(f'  {kind} MDS Kruskal stress-1: {kruskal:.4f}')
    return coords

# ── Plotting ──────────────────────────────────────────────────────────────────

tab20 = plt.colormaps['tab20']
_all_dirs  = sorted({m[2] for m in MODELS if m[2] is not None})
_dir_color = {d: tab20(i / max(len(_all_dirs), 1)) for i, d in enumerate(_all_dirs)}

_MARKERS = {'single': 'D', 'multi': 's', 'pca': '^', 'none': 'P'}

def _color(abl_dir):
    return 'black' if abl_dir is None else _dir_color[abl_dir]


def plot_mds(coords: np.ndarray, title: str, out_stem: Path):
    fig, ax = plt.subplots(figsize=(10, 8))

    for (_, role, abl_dir, stype), (x, y) in zip(MODELS, coords):
        c = _color(abl_dir)
        if role == 'teacher':
            ax.scatter(x, y, s=400, marker='*', c=[c],
                       edgecolors='black', linewidths=1.5, zorder=4)
        else:
            mk = _MARKERS.get(stype, 'D')
            ax.scatter(x, y, s=130, marker=mk, c=[c],
                       edgecolors='black', linewidths=0.6, alpha=0.85, zorder=3)

    xs, ys = coords[:, 0], coords[:, 1]
    xpad = 0.08 * (xs.max() - xs.min() or 1)
    ypad = 0.08 * (ys.max() - ys.min() or 1)
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    ax.set_xlabel('MDS 1 (a.u.)', fontsize=18)
    ax.set_ylabel('MDS 2 (a.u.)', fontsize=18)
    ax.set_title(title, fontweight='bold', fontsize=20)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_stem)
    plt.close(fig)
    print(f'  Saved: {out_stem}.pdf / .png')


def save_shared_legends():
    """Save role and direction legends once — shared across all three MDS plots."""
    role_handles = [
        Line2D([0],[0], marker='*', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=14, label='Teacher'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=9,  label='Student (single dir)'),
        Line2D([0],[0], marker='P', color='w', markerfacecolor='grey',
               markeredgecolor='black', markersize=9,  label='Student (no ablation)'),
    ]
    dir_handles = [
        mpatches.Patch(facecolor='black', edgecolor='black', label='Healthy'),
    ] + [
        mpatches.Patch(facecolor=_dir_color[d], edgecolor='black', label=f'Dir {d:02d}')
        for d in _all_dirs
    ]
    save_legend(role_handles, [h.get_label() for h in role_handles],
                OUTPUT_DIR / 'legend_role')
    save_legend(dir_handles,  [h.get_label() for h in dir_handles],
                OUTPUT_DIR / 'legend_dir', ncol=2)
    print(f'  Saved shared legends to {OUTPUT_DIR}/')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['sw','procrustes','feature','all'], default='all')
    args = ap.parse_args()

    kinds = ['sw','procrustes','feature'] if args.kind == 'all' else [args.kind]

    for kind in kinds:
        print(f'\n=== {kind.upper()} ===')
        if kind == 'sw':
            D      = load_sw()
            coords = fit_mds(D, metric=True)
            title  = 'MDS (Wasserstein)'
            plot_mds(coords, title, OUTPUT_DIR / f'mds_{kind}')
        elif kind == 'procrustes':
            D = load_procrustes()
            if np.isnan(D).any():
                print('  WARNING: procrustes matrix has NaNs (missing models) — skipping procrustes MDS')
                continue
            coords = fit_mds(D, metric=True)
            title  = 'MDS (Procrustes)'
            plot_mds(coords, title, OUTPUT_DIR / f'mds_{kind}')
        else:
            save_feature_stats()
            D = load_feature()
            for met, suffix in [(True, '_metric'), (False, '')]:
                coords = fit_mds(D, metric=met)
                label  = 'metric' if met else 'non-metric'
                title  = f'MDS (Feature, {label})'
                plot_mds(coords, title, OUTPUT_DIR / f'mds_{kind}{suffix}')

    print('\n=== Legends ===')
    save_shared_legends()
    print(f'\nAll outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
