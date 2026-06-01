#!/usr/bin/env python3
"""Regenerate timeline stats and ring geometry plots from timeline_raw_states.npz."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.vis.style import set_publication_style, save_figure

set_publication_style()
import matplotlib.pyplot as plt  # import after style

RESULTS = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'
N_BINS = 12
BIN_SZ = 360.0 / N_BINS
SPAN_COLS = ['#cde5ff', '#e8f3ff', '#dff7e2', '#f9f2d0', '#f5e4ff']


# ── helpers ──────────────────────────────────────────────────────────────────

def _bangle(a, sz=BIN_SZ):
    return int(a // sz) % N_BINS


def _avg_role(states_t, meta, cue, role):
    mask = meta[:, 0].astype(int) == cue
    st, mt = states_t[mask], meta[mask]
    bins = {b: [] for b in range(N_BINS)}
    for s, (_, c1, c2) in zip(st, mt):
        a = (c1 if cue == 1 else c2) if role == 'target' else (c2 if cue == 1 else c1)
        bins[_bangle(a)].append(s)
    out = [np.mean(bins[b], 0) for b in range(N_BINS) if bins[b]]
    return np.array(out, np.float32) if out else np.empty((0, states_t.shape[1]), np.float32)


def _ts_metrics(states_t, meta, cue):
    ta = _avg_role(states_t, meta, cue, 'target')
    da = _avg_role(states_t, meta, cue, 'distractor')
    nan = dict(plane_angle_deg=np.nan, centroid_separation=np.nan,
               target_var_pc12=np.nan, distractor_var_pc12=np.nan,
               combined_var_pc12=np.nan,
               target_planarity=np.nan, distractor_planarity=np.nan,
               target_centroid_norm=np.nan, distractor_centroid_norm=np.nan,
               target_mean_radius=np.nan, target_radius_std=np.nan,
               distractor_mean_radius=np.nan, distractor_radius_std=np.nan,
               target_eccentricity=np.nan, distractor_eccentricity=np.nan,
               target_arc_std=np.nan, distractor_arc_std=np.nan,
               ring_separation_dprime=np.nan,
               overall_mean_radius=np.nan, overall_radius_std=np.nan,
               overall_eccentricity=np.nan, overall_arc_std=np.nan,
               ring_planarity=np.nan)
    if ta.shape[0] < 3 or da.shape[0] < 3:
        return nan

    comb = np.vstack([ta, da])
    pc = PCA(n_components=3)
    coords = pc.fit_transform(comb)
    t_pts, d_pts = coords[:ta.shape[0]], coords[ta.shape[0]:]

    def _fit(pts):
        ctr = pts.mean(0)
        p = PCA(n_components=3).fit(pts - ctr)
        n = p.components_[2]; n /= np.linalg.norm(n)
        return n, ctr, float(p.explained_variance_ratio_[:2].sum())

    tn, tc, tpl = _fit(t_pts)
    dn, dc, dpl = _fit(d_pts)
    theta = float(np.degrees(np.arccos(np.clip(abs(np.dot(tn, dn)), 0, 1))))

    t_d = np.linalg.norm(t_pts - tc, axis=1)
    d_d = np.linalg.norm(d_pts - dc, axis=1)

    def _ecc(pts):
        ev = PCA(n_components=2).fit(pts).explained_variance_
        return float(ev[0] / ev[1]) if ev[1] > 1e-12 else np.nan

    def _arc(pts):
        arcs = [np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts))]
        return float(np.std(arcs))

    def _vpc12(x):
        if x.shape[0] < 3: return np.nan
        return float(PCA(n_components=min(3, x.shape[0], x.shape[1])).fit(x).explained_variance_ratio_[:2].sum())

    t_mr, d_mr = float(t_d.mean()), float(d_d.mean())
    t_rs, d_rs = float(t_d.std()),  float(d_d.std())
    denom = np.sqrt(0.5 * (t_mr**2 + d_mr**2))
    dprime = float(np.linalg.norm(tc - dc) / denom) if denom > 0 else np.nan

    return dict(
        plane_angle_deg=theta,
        centroid_separation=float(np.linalg.norm(tc - dc)),
        target_var_pc12=_vpc12(ta), distractor_var_pc12=_vpc12(da),
        combined_var_pc12=float(pc.explained_variance_ratio_[:2].sum()),
        target_planarity=tpl, distractor_planarity=dpl,
        target_centroid_norm=float(np.linalg.norm(tc)),
        distractor_centroid_norm=float(np.linalg.norm(dc)),
        target_mean_radius=t_mr, target_radius_std=t_rs,
        distractor_mean_radius=d_mr, distractor_radius_std=d_rs,
        target_eccentricity=_ecc(t_pts), distractor_eccentricity=_ecc(d_pts),
        target_arc_std=_arc(t_pts), distractor_arc_std=_arc(d_pts),
        ring_separation_dprime=dprime,
        overall_mean_radius=0.5*(t_mr+d_mr), overall_radius_std=0.5*(t_rs+d_rs),
        overall_eccentricity=0.5*(_ecc(t_pts)+_ecc(d_pts)),
        overall_arc_std=0.5*(_arc(t_pts)+_arc(d_pts)),
        ring_planarity=0.5*(tpl+dpl),
    )


def _add_phase_markers(ax, ph_bounds, ph_names, total):
    for i in range(len(ph_names)):
        ax.axvspan(ph_bounds[i] - .5, ph_bounds[i+1] - .5,
                   color=SPAN_COLS[i % len(SPAN_COLS)], alpha=0.18, zorder=0)
    for b in ph_bounds[1:-1]:
        ax.axvline(b - .5, color='k', linestyle='--', linewidth=.9, alpha=.45)
    ctrs = [(ph_bounds[i] + ph_bounds[i+1] - 1) / 2 for i in range(len(ph_names))]
    sx = ax.secondary_xaxis('top')
    sx.set_xticks(ctrs)
    sx.set_xticklabels(ph_names, fontsize=8)
    ax.set_xlim(-.5, total - .5)


# ── main per-directory function ───────────────────────────────────────────────

def replot_dir(d):
    data = np.load(d / 'timeline_raw_states.npz', allow_pickle=True)
    tl_st = data['tl_st']   # (n_trials, T, D)
    tl_mt = data['tl_mt']   # (n_trials, 3)

    if 'pep_ref' in data:
        pep_ref = [int(x) for x in data['pep_ref']]
        n_diff = int(data['n_diffusion_steps'])
    else:
        # Older format: only prep steps saved, use known default split
        if tl_st.shape[1] == 26:
            pep_ref = [5, 15, 5, 1]
        else:
            pep_ref = [tl_st.shape[1]]
        n_diff = 0

    prep_len = sum(pep_ref)
    TOTAL = int(tl_st.shape[1])

    ph_names = [f'Prep {i}' for i in range(len(pep_ref))] + (['Diffusion'] if n_diff > 0 else [])
    ph_bounds = [0]
    cur = 0
    for dur in pep_ref:
        cur += dur; ph_bounds.append(cur)
    if n_diff > 0:
        ph_bounds.append(TOTAL)

    # Compute metrics for each timestep
    x = np.arange(TOTAL)
    mets = {c: [_ts_metrics(tl_st[:, t, :], tl_mt, c) for t in range(TOTAL)] for c in [1, 2]}

    def _pm(ax):
        _add_phase_markers(ax, ph_bounds, ph_names, TOTAL)

    # ── Figure 1: Cue 1 vs Cue 2 (timeline stats) ────────────────────────────
    c1a = np.array([m['plane_angle_deg']    for m in mets[1]])
    c2a = np.array([m['plane_angle_deg']    for m in mets[2]])
    c1s = np.array([m['centroid_separation'] for m in mets[1]])
    c2s = np.array([m['centroid_separation'] for m in mets[2]])
    c1v = np.array([m['combined_var_pc12']   for m in mets[1]])
    c2v = np.array([m['combined_var_pc12']   for m in mets[2]])
    c1p = np.array([m['target_planarity']    for m in mets[1]])
    c2p = np.array([m['target_planarity']    for m in mets[2]])

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle('Cue 1 vs Cue 2', fontweight='bold')
    specs = [
        (0, 0, c1a,  c2a,  'Plane angle',       'Angle (deg)'),
        (0, 1, c1p,  c2p,  'Planarity',         'Planarity (a.u.)'),
        (1, 0, c1s,  c2s,  'Centroid separation','Separation (a.u.)'),
        (1, 1, c1v,  c2v,  'PCA variance PC1+PC2','Var. explained (a.u.)'),
    ]
    for r, c, a1, a2, ttl, yl in specs:
        ax = axes[r, c]
        ax.plot(x, a1, color='tab:red',  lw=2, label='Cue 1')
        ax.plot(x, a2, color='tab:blue', lw=2, label='Cue 2')
        ax.set_title(ttl); ax.set_xlabel('Step'); ax.set_ylabel(yl)
        ax.legend(); _pm(ax)
    fig.tight_layout()
    save_figure(fig, d / 'timeline_stats_cue1_vs_cue2')
    plt.close(fig)

    # ── Figure 2: Ring geometry (Cue 1 vs Cue 2) ─────────────────────────────
    c1_omr = np.array([m['overall_mean_radius']    for m in mets[1]])
    c2_omr = np.array([m['overall_mean_radius']    for m in mets[2]])
    c1_ors = np.array([m['overall_radius_std']     for m in mets[1]])
    c2_ors = np.array([m['overall_radius_std']     for m in mets[2]])
    c1_oe  = np.array([m['overall_eccentricity']   for m in mets[1]])
    c2_oe  = np.array([m['overall_eccentricity']   for m in mets[2]])
    c1_oa  = np.array([m['overall_arc_std']        for m in mets[1]])
    c2_oa  = np.array([m['overall_arc_std']        for m in mets[2]])
    c1_rp  = np.array([m['ring_planarity']         for m in mets[1]])
    c2_rp  = np.array([m['ring_planarity']         for m in mets[2]])
    c1_dp  = np.array([m['ring_separation_dprime'] for m in mets[1]])
    c2_dp  = np.array([m['ring_separation_dprime'] for m in mets[2]])

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle('Ring geometry: Cue 1 vs Cue 2', fontweight='bold')
    specs = [
        (0, 0, c1_omr, c2_omr, 'Mean ring radius',     'Radius (a.u.)'),
        (0, 1, c1_ors, c2_ors, 'Ring radius std',       'Std (a.u.)'),
        (0, 2, c1_oe,  c2_oe,  'Ring eccentricity',     'PC1/PC2 ratio'),
        (1, 0, c1_oa,  c2_oa,  'Arc-spacing std',       'Arc std (a.u.)'),
        (1, 1, c1_rp,  c2_rp,  'Ring planarity',        'Var. explained (a.u.)'),
        (1, 2, c1_dp,  c2_dp,  "Ring separation $d'$",  "$d'$"),
    ]
    for r, c, a1, a2, ttl, yl in specs:
        ax = axes[r, c]
        ax.plot(x, a1, color='tab:red',  lw=2, label='Cue 1')
        ax.plot(x, a2, color='tab:blue', lw=2, label='Cue 2')
        ax.set_title(ttl); ax.set_xlabel('Step'); ax.set_ylabel(yl)
        ax.legend(); _pm(ax)
    fig.tight_layout()
    save_figure(fig, d / 'timeline_ring_geometry_cue1_vs_cue2')
    plt.close(fig)

    # ── Figures 3 & 4: per-cue target vs distractor ──────────────────────────
    for cue, ct, cd in [(1, 'tab:green', 'tab:orange'), (2, 'tab:purple', 'tab:brown')]:
        vt  = np.array([m['target_var_pc12']         for m in mets[cue]])
        vd  = np.array([m['distractor_var_pc12']     for m in mets[cue]])
        pt  = np.array([m['target_planarity']        for m in mets[cue]])
        pd_ = np.array([m['distractor_planarity']    for m in mets[cue]])
        nt  = np.array([m['target_centroid_norm']    for m in mets[cue]])
        nd  = np.array([m['distractor_centroid_norm']for m in mets[cue]])
        ag  = np.array([m['plane_angle_deg']         for m in mets[cue]])

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle(f'Cue {cue}: target vs distractor', fontweight='bold')
        specs = [
            (0, 0, ag,  None, None,  None,  'Plane angle',       'Angle (deg)'),
            (0, 1, pt,  pd_,  'Target', 'Distractor', 'Planarity',         'Planarity (a.u.)'),
            (1, 0, nt,  nd,   'Target centroid', 'Distractor centroid', 'Centroid norm', 'Norm (a.u.)'),
            (1, 1, vt,  vd,   'Target', 'Distractor', 'PCA variance PC1+PC2', 'Var. explained (a.u.)'),
        ]
        for r, c, a1, a2, l1, l2, ttl, yl in specs:
            ax = axes[r, c]
            ax.plot(x, a1, color=ct, lw=2, label=l1 or f'Cue {cue}')
            if a2 is not None:
                ax.plot(x, a2, color=cd, lw=2, label=l2)
            ax.set_title(ttl); ax.set_xlabel('Step'); ax.set_ylabel(yl)
            ax.legend(); _pm(ax)
        fig.tight_layout()
        save_figure(fig, d / f'timeline_stats_cue{cue}_target_vs_distractor')
        plt.close(fig)

        # Ring geometry per cue
        tmr = np.array([m['target_mean_radius']      for m in mets[cue]])
        dmr = np.array([m['distractor_mean_radius']  for m in mets[cue]])
        trs = np.array([m['target_radius_std']       for m in mets[cue]])
        drs = np.array([m['distractor_radius_std']   for m in mets[cue]])
        te  = np.array([m['target_eccentricity']     for m in mets[cue]])
        de  = np.array([m['distractor_eccentricity'] for m in mets[cue]])
        ta  = np.array([m['target_arc_std']          for m in mets[cue]])
        da  = np.array([m['distractor_arc_std']      for m in mets[cue]])

        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle(f'Ring geometry cue {cue}: target vs distractor', fontweight='bold')
        specs = [
            (0, 0, tmr, dmr, 'Target radius',  'Distractor radius',  'Mean ring radius',  'Radius (a.u.)'),
            (0, 1, trs, drs, 'Target std',     'Distractor std',     'Ring radius std',   'Std (a.u.)'),
            (1, 0, te,  de,  'Target',         'Distractor',         'Ring eccentricity', 'PC1/PC2 ratio'),
            (1, 1, ta,  da,  'Target arc std', 'Distractor arc std', 'Arc-spacing std',   'Arc std (a.u.)'),
        ]
        for r, c, a1, a2, l1, l2, ttl, yl in specs:
            ax = axes[r, c]
            ax.plot(x, a1, color=ct, lw=2, label=l1)
            ax.plot(x, a2, color=cd, lw=2, label=l2)
            ax.set_title(ttl); ax.set_xlabel('Step'); ax.set_ylabel(yl)
            ax.legend(); _pm(ax)
        fig.tight_layout()
        save_figure(fig, d / f'timeline_ring_geometry_cue{cue}_target_vs_distractor')
        plt.close(fig)


if __name__ == '__main__':
    dirs = sorted(
        d for d in RESULTS.iterdir()
        if d.is_dir() and (d / 'timeline_raw_states.npz').exists()
    )
    print(f'Found {len(dirs)} directories')
    for i, d in enumerate(dirs, 1):
        print(f'[{i}/{len(dirs)}] {d.name}')
        replot_dir(d)
        print(f'  Done.')
    print('All done.')
