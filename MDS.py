#!/usr/bin/env python3
"""
MDS projection of teacher + student space using hand-crafted and topological features.
Computes difference + sum modes for all metrics, ranks by variance, visualizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import re
import sys

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_DIR = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'

N_BINS = 12
BIN_SIZE = 360.0 / N_BINS
DEVICE = 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def _bangle(a, sz=30.0):
    return int(a // sz) % int(360 // sz)

def _group_bins(tl_mt):
    """Group trial indices by cue and target/distractor angle."""
    g = {c: {'target': {b: [] for b in range(N_BINS)},
              'distractor': {b: [] for b in range(N_BINS)}} for c in [1, 2]}
    for i, (cue, c1, c2) in enumerate(tl_mt):
        cue = int(cue)
        ta, da = (c1, c2) if cue == 1 else (c2, c1)
        g[cue]['target'][int(ta // BIN_SIZE) % N_BINS].append(i)
        g[cue]['distractor'][int(da // BIN_SIZE) % N_BINS].append(i)
    for c in [1, 2]:
        for role in ['target', 'distractor']:
            for b in range(N_BINS):
                g[c][role][b] = np.array(g[c][role][b], dtype=np.int64)
    return g

def _bin_avg(tl_st, idx_per_bin):
    """Average neural states within each angle bin."""
    T, D = tl_st.shape[1], tl_st.shape[2]
    out = np.zeros((N_BINS, T, D), np.float32)
    for b, idx in idx_per_bin.items():
        if len(idx):
            out[b] = tl_st[idx].mean(0)
    return out

def _ring_metrics_over_time(tgt_bm, dis_bm):
    """Compute ring geometry + separation metrics over time."""
    T = tgt_bm.shape[1]
    mr   = np.full((2, T), np.nan, np.float32)
    rs   = np.full((2, T), np.nan, np.float32)
    ecc  = np.full((2, T), np.nan, np.float32)
    arc  = np.full((2, T), np.nan, np.float32)
    plan = np.full((2, T), np.nan, np.float32)
    cnrm = np.full((2, T), np.nan, np.float32)
    vpc  = np.full((2, T), np.nan, np.float32)
    pang = np.full(T, np.nan, np.float32)
    csep = np.full(T, np.nan, np.float32)
    vcom = np.full(T, np.nan, np.float32)

    for t in range(T):
        tp, dp = tgt_bm[:, t, :], dis_bm[:, t, :]
        comb   = np.vstack([tp, dp])
        cpca   = PCA(n_components=3)
        coords = cpca.fit_transform(comb)
        t3, d3 = coords[:N_BINS], coords[N_BINS:]
        vcom[t] = float(cpca.explained_variance_ratio_[:2].sum())

        tc, dc = t3.mean(0), d3.mean(0)
        td = np.linalg.norm(t3 - tc, axis=1)
        dd = np.linalg.norm(d3 - dc, axis=1)
        mr[0, t], mr[1, t]   = td.mean(), dd.mean()
        rs[0, t], rs[1, t]   = td.std(),  dd.std()
        cnrm[0, t]            = np.linalg.norm(tc)
        cnrm[1, t]            = np.linalg.norm(dc)
        csep[t]               = np.linalg.norm(tc - dc)

        nt = nd = None
        for ri, (pts, ctr) in enumerate([(t3, tc), (d3, dc)]):
            p = PCA(n_components=3).fit(pts - ctr)
            plan[ri, t] = float(p.explained_variance_ratio_[:2].sum())
            vpc[ri, t]  = float(PCA(n_components=2).fit(pts).explained_variance_ratio_.sum())
            if ri == 0: nt = p.components_[2]
            else:       nd = p.components_[2]

        pang[t] = float(np.degrees(np.arccos(np.clip(abs(np.dot(nt, nd)), 0, 1))))

        def _ecc(pts):
            ev = PCA(n_components=2).fit(pts).explained_variance_
            return ev[0] / ev[1] if ev[1] > 1e-12 else np.nan
        ecc[0, t], ecc[1, t] = _ecc(t3), _ecc(d3)

        def _arc(pts):
            return float(np.std([np.linalg.norm(pts[(i+1)%N_BINS]-pts[i]) for i in range(N_BINS)]))
        arc[0, t], arc[1, t] = _arc(t3), _arc(d3)

    return {'mean_radius': mr, 'radius_std': rs, 'eccentricity': ecc, 'arc_std': arc,
            'planarity': plan, 'centroid_norm': cnrm, 'var_pc12': vpc,
            'plane_angle_deg': pang, 'centroid_separation': csep, 'var_pc12_combined': vcom}

# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_topological_features(tl_st, per_cue=False):
    """Compute topological features from neural states (properly normalized)."""
    features = {}
    X = tl_st.reshape(-1, tl_st.shape[-1]).astype(np.float32)

    # 1. Intrinsic dimensionality (% variance at 90%)
    pca = PCA()
    pca.fit(X)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    id_90 = float(np.argmax(cumsum_var >= 0.9) + 1) / len(pca.explained_variance_ratio_)
    features['intrinsic_dim_90pct'] = id_90

    # 2. Participation ratio
    lam = pca.explained_variance_
    part_ratio = float((lam.sum() ** 2) / (lam ** 2).sum())
    features['participation_ratio'] = part_ratio

    # 3. Variance explained by top-3 PCs
    var_top3 = float(pca.explained_variance_ratio_[:3].sum())
    features['var_top3_pcs'] = var_top3

    # 4. Reconstruction error at rank-3 (NORMALIZED per-sample)
    X_centered = X - X.mean(0)
    pca3 = PCA(n_components=3)
    X_recon = pca3.inverse_transform(pca3.fit_transform(X_centered))
    per_sample_err = np.sum((X_centered - X_recon) ** 2, axis=1)
    recon_err_normalized = float(np.sqrt(per_sample_err.mean()))
    features['reconstruction_error_rank3'] = recon_err_normalized

    # 5. Mean squared temporal derivative (per-trial average, normalized)
    deriv_per_trial = []
    for trial in tl_st:
        diffs = np.diff(trial, axis=0)
        squared_norms = np.sum(diffs ** 2, axis=1)
        mean_deriv = np.mean(squared_norms)
        deriv_per_trial.append(mean_deriv)
    mstd = float(np.mean(deriv_per_trial))
    features['mean_sq_temporal_deriv'] = mstd

    # 6. Principal angle: target vs distractor subspaces
    tgt_states = X[::2] if len(X) % 2 == 0 else X[:-1][::2]
    dis_states = X[1::2] if len(X) % 2 == 0 else X[1:][1::2]
    if len(tgt_states) > 3 and len(dis_states) > 3:
        pca_tgt = PCA(n_components=3)
        pca_dis = PCA(n_components=3)
        pca_tgt.fit(tgt_states)
        pca_dis.fit(dis_states)
        cos_angle = abs(np.dot(pca_tgt.components_[0], pca_dis.components_[0]))
        principal_angle = float(np.degrees(np.arccos(np.clip(cos_angle, 0, 1))))
        features['principal_angle_tgt_dis'] = principal_angle
    else:
        features['principal_angle_tgt_dis'] = 0.0

    # 7. Radius of curvature (normalized)
    curvatures = []
    for trial in tl_st:
        if trial.shape[0] > 2:
            diffs1 = np.diff(trial, axis=0)
            diffs2 = np.diff(diffs1, axis=0)
            if len(diffs2) > 0:
                speeds = np.linalg.norm(diffs1[:-1], axis=1) + 1e-10
                curvatures_trial = np.linalg.norm(diffs2, axis=1) / speeds
                curvatures.append(np.mean(curvatures_trial))
    features['radius_curvature'] = float(np.mean(curvatures)) if curvatures else 0.0

    # 8. Signal-to-noise ratio (between-trial vs within-trial variance)
    snr_vals = []
    for pc_idx in range(min(3, len(pca.components_))):
        pc = pca.components_[pc_idx]
        proj = X @ pc
        proj_reshaped = proj.reshape(tl_st.shape[0], tl_st.shape[1])
        between_trial_var = np.var(proj_reshaped.mean(axis=1))
        within_trial_var = np.mean(np.var(proj_reshaped, axis=1))
        snr = between_trial_var / (within_trial_var + 1e-10)
        snr_vals.append(snr)
    features['snr_pc_avg'] = float(np.mean(snr_vals)) if snr_vals else 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(npz_path):
    """
    Extract all features (hand-crafted + topological, with diff/sum modes).

    Returns:
        dict: feature_name -> value (averaged across cues where applicable)
    """
    npz = np.load(npz_path)
    tl_st = npz['tl_st']  # (n_trials, T, D)
    tl_mt = npz['tl_mt']  # (n_trials, 3)

    features = {}

    # ─── Hand-crafted metrics (per cue) ────────────────────────────────────
    g = _group_bins(tl_mt)
    metrics_per_cue = {}

    for c in [1, 2]:
        tgt_bm = _bin_avg(tl_st, g[c]['target'])
        dis_bm = _bin_avg(tl_st, g[c]['distractor'])
        metrics_per_cue[c] = _ring_metrics_over_time(tgt_bm, dis_bm)

    # Extract and compute diff/sum modes for hand-crafted metrics
    metric_names = ['mean_radius', 'radius_std', 'eccentricity', 'arc_std',
                    'planarity', 'centroid_norm', 'var_pc12',
                    'plane_angle_deg', 'centroid_separation', 'var_pc12_combined']

    for m in metric_names:
        m1 = metrics_per_cue[1][m]
        m2 = metrics_per_cue[2][m]

        # Average across time (and target/distractor if 2D)
        if m1.ndim == 2:  # (2, T) for tgt-dis metrics
            v1 = m1.mean()
            v2 = m2.mean()
        else:  # (T,) for separation metrics
            v1 = m1.mean()
            v2 = m2.mean()

        features[f'{m}_diff'] = v1 - v2
        features[f'{m}_sum'] = v1 + v2

    # ─── Topological features ─────────────────────────────────────────────
    topo = compute_topological_features(tl_st)

    # Add diff/sum for topological features too (compute per cue-grouped data)
    topo_vals = {}
    for c in [1, 2]:
        tl_st_c = tl_st[tl_mt[:, 0] == c]  # states for this cue
        topo_c = compute_topological_features(tl_st_c)
        topo_vals[c] = topo_c

    for topo_name in topo.keys():
        v1 = topo_vals[1][topo_name]
        v2 = topo_vals[2][topo_name]
        features[f'{topo_name}_diff'] = v1 - v2
        features[f'{topo_name}_sum'] = v1 + v2

    return features

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("MDS ANALYSIS: Teacher + Student Space")
    print("=" * 80)

    # ─── Load all NPZs ────────────────────────────────────────────────────
    print("\nScanning NPZ files...")
    teachers = {}
    students = {}

    for d in RESULTS_DIR.iterdir():
        npz = d / 'timeline_raw_states.npz'
        if not npz.exists():
            continue

        # Parse run name
        m_teacher = re.match(r'^ablated_teacher_dir_(\d+)$', d.name)
        if m_teacher:
            teacher_id = int(m_teacher.group(1))
            if teacher_id in [2, 3, 9, 11]:  # Skip ablation directions 2, 3, 9, 11
                continue
            teachers[teacher_id] = npz
            continue

        if d.name == 'index_cued_first_diffusion_0.3_swap_7':
            teachers['no_ablation'] = npz
            continue

        m_student = re.match(r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(.+)_(\d+)$', d.name)
        if m_student:
            key = m_student.group(1)
            try:
                key = int(key)
            except ValueError:
                pass
            if key != 'idk' and key not in [2, 3, 9, 11]:  # Skip ablation directions 2, 3, 9, 11
                students.setdefault(key, []).append((d.name, npz))

    print(f"  Teachers: {len(teachers)}")
    print(f"  Students: {sum(len(v) for v in students.values())}")

    # ─── Extract features ─────────────────────────────────────────────────
    print("\nExtracting features...")
    feature_dict = {}

    for tk, tnpz in sorted(teachers.items(), key=str):
        print(f"  teacher {tk}...", end=' ', flush=True)
        feature_dict[('teacher', tk)] = extract_all_features(tnpz)
        print("OK")

    for sk, stu_list in sorted(students.items(), key=str):
        for sname, snpz in stu_list:
            feature_dict[('student', sk, sname)] = extract_all_features(snpz)

    print(f"Extracted {len(feature_dict)} feature vectors")

    # ─── Feature selection by variance ────────────────────────────────────
    print("\nFeature selection...")

    # Use all features
    all_feat_names = list(feature_dict[('teacher', list(teachers.keys())[0])].keys())
    print(f"  Total features: {len(all_feat_names)}")

    # Compute variance across teachers
    teacher_features = np.array([[feature_dict[('teacher', tk)][fn]
                                  for fn in all_feat_names]
                                 for tk in sorted(teachers.keys(), key=str)])
    feat_vars = teacher_features.var(axis=0)

    # Select top K (e.g., 10-15)
    top_k = min(15, len(all_feat_names))
    top_k_idx = np.argsort(feat_vars)[-top_k:][::-1]
    top_k_names = [all_feat_names[i] for i in top_k_idx]

    print(f"\n  Top {top_k} features by variance:")
    for i, idx in enumerate(top_k_idx):
        print(f"    {i+1}. {all_feat_names[idx]:40s} var={feat_vars[idx]:.6f}")

    # ─── Build feature matrix (selected features only) ────────────────────
    print("\nBuilding feature matrix...")

    all_keys = ([(('teacher', tk), tk) for tk in sorted(teachers.keys(), key=str)] +
                [(('student', sk, sname), f"{sk}_{sname}")
                 for sk in sorted(students.keys(), key=str)
                 for sname, _ in sorted(students[sk], key=lambda x: x[0])])

    n_all = len(all_keys)
    fv_matrix = np.zeros((n_all, top_k), dtype=np.float32)

    for i, (key, _) in enumerate(all_keys):
        for j, fname in enumerate(top_k_names):
            fv_matrix[i, j] = feature_dict[key][fname]

    # Standardize
    scaler = StandardScaler()
    fv_matrix_scaled = scaler.fit_transform(fv_matrix)

    # ─── Distance matrix ──────────────────────────────────────────────────
    print("Computing distance matrix...")
    dist_vec = pdist(fv_matrix_scaled, metric='euclidean')
    dist_matrix = squareform(dist_vec)

    print(f"  Distance matrix: {dist_matrix.shape}")
    print(f"  Min: {dist_matrix[dist_matrix > 0].min():.3f}, Max: {dist_matrix.max():.3f}")

    # ─── MDS projection ───────────────────────────────────────────────────
    print("\nFitting MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(dist_matrix)
    print(f"  Stress: {mds.stress_:.4f}")

    # ─── Visualization ────────────────────────────────────────────────────
    print("\nPlotting...")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Teachers
    teacher_indices = [i for i, (k, _) in enumerate(all_keys) if k[0] == 'teacher']
    teacher_coords = mds_coords[teacher_indices]
    teacher_labels = [str(k[1]) for k, _ in [all_keys[i] for i in teacher_indices]]

    scatter_t = ax.scatter(teacher_coords[:, 0], teacher_coords[:, 1],
                          s=400, marker='o', c=range(len(teacher_coords)),
                          cmap='tab20', edgecolors='black', lw=2.5, label='Teachers', zorder=3)

    for i, (xi, yi) in enumerate(teacher_coords):
        ax.text(xi, yi, f"T{teacher_labels[i]}", ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=4)

    # Students (colored by teacher ablation direction)
    student_indices = [i for i, (k, _) in enumerate(all_keys) if k[0] == 'student']
    student_coords = mds_coords[student_indices]
    student_keys = [all_keys[i][0] for i in student_indices]
    student_teacher_dirs = [k[1] for k in student_keys]  # ablation direction (key)

    # Map teacher directions to colors
    unique_dirs = sorted(set(student_teacher_dirs), key=str)
    dir_to_color_idx = {d: i for i, d in enumerate(unique_dirs)}
    student_colors = [dir_to_color_idx[d] for d in student_teacher_dirs]

    scatter_s = ax.scatter(student_coords[:, 0], student_coords[:, 1],
                          s=120, marker='x', c=student_colors, cmap='tab20',
                          alpha=0.8, lw=2, label='Students', zorder=2)

    # Add student labels (abbreviated)
    for i, (xi, yi) in enumerate(student_coords):
        teacher_dir = student_teacher_dirs[i]
        ax.text(xi, yi, f"S{teacher_dir}", ha='center', va='center',
               fontsize=7, alpha=0.6, zorder=2)

    ax.set_xlabel('MDS 1', fontsize=12)
    ax.set_ylabel('MDS 2', fontsize=12)
    ax.set_title(f'Teacher/Student Space (MDS, top {top_k} features by variance)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, zorder=0)

    plt.tight_layout()
    output_path = RESULTS_DIR / 'MDS_teacher_student_space.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()

    print("\n" + "=" * 80)
    print(f"✓ Done! {len(teacher_indices)} teachers, {len(student_indices)} students")
    print("=" * 80)

if __name__ == '__main__':
    main()
