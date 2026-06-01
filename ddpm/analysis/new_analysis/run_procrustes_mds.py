#!/usr/bin/env python3
"""
Build a full N×N pairwise Procrustes distance matrix from three precomputed blocks
(teacher-teacher, student-student, teacher-student) then run metric MDS and plot.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.manifold import MDS
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.vis.style import set_publication_style, save_figure, save_legend

set_publication_style()
import matplotlib.pyplot as plt

RESULTS = REPO_ROOT / 'ddpm/analysis/new_analysis/results'
OUTPUT_DIR = RESULTS / 'procrustes_mds'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load blocks ───────────────────────────────────────────────────────────────
tt_mat   = np.load(RESULTS / 'teacher_teacher_heatmap/heatmap_fixed.npy')
tt_labels_raw = np.load(RESULTS / 'teacher_teacher_heatmap/labels.npz', allow_pickle=True)
tt_labels = [str(x) for x in tt_labels_raw.get('labels', tt_labels_raw.get('teacher_labels'))]

ss_mat   = np.load(RESULTS / 'student_student_heatmap/heatmap_fixed.npy')
ss_labels_raw = np.load(RESULTS / 'student_student_heatmap/labels.npz', allow_pickle=True)
ss_labels = [str(x) for x in ss_labels_raw.get('labels', ss_labels_raw.get('student_labels'))]

ts_mat   = np.load(RESULTS / 'procrustes_heatmap_15dirs_all_students/heatmap_fixed.npy')
ts_labels_raw = np.load(RESULTS / 'procrustes_heatmap_15dirs_all_students/labels.npz', allow_pickle=True)
ts_t_labels = [str(x) for x in ts_labels_raw['teacher_labels']]
ts_s_labels = [str(x) for x in ts_labels_raw['student_labels']]

print(f'Loaded blocks: TT={tt_mat.shape}, SS={ss_mat.shape}, TS={ts_mat.shape}')

# ── Normalize student labels: abl{d}_s{seed} → abl{d}_{seed}, no_abl → noabl_s0
def _norm_ss_label(lab):
    if lab == 'no_abl':
        return 'noabl_s0'
    # abl{d}_s{seed} → abl{d}_{seed}
    import re
    m = re.match(r'^abl(\d+)_s(\d+)$', lab)
    if m:
        return f'abl{m.group(1)}_{m.group(2)}'
    return lab

ss_labels_norm = [_norm_ss_label(l) for l in ss_labels]

# ── Exclusions (consistent across all MDS scripts) ───────────────────────────
EXCLUDE_TEACHER = {'dir_00', 'dir_02', 'dir_03', 'dir_06', 'dir_08',
                   'dir_09', 'dir_10', 'dir_11'}
EXCLUDE_STUDENT = {'abl0_0', 'abl0_1', 'abl2_0', 'abl3_0', 'abl6_0',
                   'abl9_0', 'abl10_0', 'abl11_0', 'abl11_1', 'multi_0-1_s0'}

# ── Build included model lists ────────────────────────────────────────────────
# Teachers: must appear in both TT and TS teacher axis
t_keep = [l for l in tt_labels if l not in EXCLUDE_TEACHER and l in ts_t_labels]

# Students: must appear in SS (normalized) and TS student axis, and not excluded
s_keep = [l for l in ss_labels_norm
          if l not in EXCLUDE_STUDENT and l in ts_s_labels]

print(f'Included: {len(t_keep)} teachers, {len(s_keep)} students  (total {len(t_keep)+len(s_keep)})')
print('  Teachers:', t_keep)
print('  Students:', s_keep)

# ── Index maps ────────────────────────────────────────────────────────────────
tt_idx = {l: i for i, l in enumerate(tt_labels)}
ss_idx = {l: i for i, l in enumerate(ss_labels_norm)}
ts_t   = {l: i for i, l in enumerate(ts_t_labels)}
ts_s   = {l: i for i, l in enumerate(ts_s_labels)}

N  = len(t_keep) + len(s_keep)
nt = len(t_keep)
ns = len(s_keep)

D = np.zeros((N, N), dtype=np.float64)

# Teacher-teacher block
for i, ti in enumerate(t_keep):
    for j, tj in enumerate(t_keep):
        D[i, j] = tt_mat[tt_idx[ti], tt_idx[tj]]

# Student-student block
for i, si in enumerate(s_keep):
    for j, sj in enumerate(s_keep):
        D[nt + i, nt + j] = ss_mat[ss_idx[si], ss_idx[sj]]

# Teacher-student + student-teacher blocks
for i, ti in enumerate(t_keep):
    for j, sj in enumerate(s_keep):
        v = ts_mat[ts_t[ti], ts_s[sj]]
        D[i,      nt + j] = v
        D[nt + j, i     ] = v

# Symmetrize (average with transpose to remove any asymmetry)
D = 0.5 * (D + D.T)
np.fill_diagonal(D, 0.0)

print(f'Distance matrix: {D.shape}  range [{D[D>0].min():.4f}, {D.max():.4f}]')

# ── MDS ───────────────────────────────────────────────────────────────────────
mds_m  = MDS(n_components=2, dissimilarity='precomputed', metric=True,  random_state=42)
coords_m  = mds_m.fit_transform(D)
print(f'Metric MDS stress:     {mds_m.stress_:.4f}')

mds_nm = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42)
coords_nm = mds_nm.fit_transform(D)
print(f'Non-metric MDS stress: {mds_nm.stress_:.4f}')

# ── Save data ─────────────────────────────────────────────────────────────────
all_labels = t_keep + s_keep
is_teacher = [True] * nt + [False] * ns
np.savez(OUTPUT_DIR / 'mds_data.npz',
         coords_metric=coords_m, coords_nonmetric=coords_nm,
         labels=np.array(all_labels), is_teacher=np.array(is_teacher),
         dist_matrix=D)
np.save(OUTPUT_DIR / 'mds_coords.npy', coords_m)

# ── Plot helpers ──────────────────────────────────────────────────────────────
tab20 = plt.colormaps['tab20']

import re as _re
def _dir(label):
    m = _re.search(r'(\d+)', label)
    return int(m.group(1)) if m else -1

def _stype(label):
    if label.startswith('multi_'): return 'multi'
    if label.startswith('pca_'):   return 'pca'
    if label == 'noabl_s0':        return 'none'
    return 'single'

_MARKERS = {'single': 'D', 'multi': 's', 'pca': '^', 'none': 'P'}

all_dirs  = sorted({_dir(l) for l in all_labels if _dir(l) >= 0})
dir_color = {d: tab20(i / max(len(all_dirs), 1)) for i, d in enumerate(all_dirs)}

def _color(label, is_t):
    d = _dir(label)
    if is_t and d < 0:
        return 'black'
    return dir_color.get(d, 'grey')

role_handles = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=14, label='Teacher'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=9,  label='Student (single dir)'),
    Line2D([0],[0], marker='P', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=9,  label='Student (no ablation)'),
]
dir_handles = [
    mpatches.Patch(facecolor='black', edgecolor='black', label='healthy'),
] + [
    mpatches.Patch(facecolor=dir_color[d], edgecolor='black', label=f'dir {d:02d}')
    for d in sorted(all_dirs) if d >= 0
]

def _plot(coords, title, out_stem):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (label, is_t) in enumerate(zip(all_labels, is_teacher)):
        x, y = coords[i]
        c  = _color(label, is_t)
        if is_t:
            ax.scatter(x, y, s=400, marker='*', c=[c],
                       edgecolors='black', linewidths=1.5, zorder=4)
        else:
            mk = _MARKERS[_stype(label)]
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
    print(f'Saved: {out_stem}.pdf / .png')

# ── Generate both plots ───────────────────────────────────────────────────────
_plot(coords_m,  'Procrustes MDS (metric)',     OUTPUT_DIR / 'mds_procrustes_metric')
_plot(coords_nm, 'Procrustes MDS (non-metric)', OUTPUT_DIR / 'mds_procrustes_nonmetric')

# Shared standalone legends (same for both plots)
save_legend(role_handles, [h.get_label() for h in role_handles],
            OUTPUT_DIR / 'mds_procrustes_role')
save_legend(dir_handles,  [h.get_label() for h in dir_handles],
            OUTPUT_DIR / 'mds_procrustes_dir', ncol=2)

print('Done.')
