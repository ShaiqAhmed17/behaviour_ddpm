#!/usr/bin/env python3
"""
Run MDS with different feature subsets ranked by importance.
Compare stress values across feature selection strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from MDS import extract_all_features

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_DIR = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'

EXCLUDE_DIRS = {2, 3, 9, 11}

# ─────────────────────────────────────────────────────────────────────────────
# Load data (same as before)
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data...")
teachers = {}
students = {}

for d in RESULTS_DIR.iterdir():
    npz = d / 'timeline_raw_states.npz'
    if not npz.exists():
        continue

    m_teacher = re.match(r'^ablated_teacher_dir_(\d+)$', d.name)
    if m_teacher:
        teacher_id = int(m_teacher.group(1))
        if teacher_id in EXCLUDE_DIRS:
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
        if key != 'idk' and key not in EXCLUDE_DIRS:
            students.setdefault(key, []).append((d.name, npz))

print(f"  Teachers: {len(teachers)}, Students: {sum(len(v) for v in students.values())}")

# Extract features
print("Extracting features...")
features_all = {}

for tk, tnpz in sorted(teachers.items(), key=str):
    features_all[('teacher', tk)] = extract_all_features(tnpz)

for sk, stu_list in sorted(students.items(), key=str):
    for sname, snpz in stu_list:
        features_all[('student', sk, sname)] = extract_all_features(snpz)

all_feat_names = sorted(list(features_all[('teacher', list(teachers.keys())[0])].keys()))
all_keys = (
    [(('teacher', tk)) for tk in sorted(teachers.keys(), key=str)] +
    [(('student', sk, sname)) for sk in sorted(students.keys(), key=str)
     for sname, _ in sorted(students[sk])]
)

feature_matrix_all = np.array([[features_all[key][fname] for fname in all_feat_names]
                               for key in all_keys])

# Read importance scores
importance_df = pd.read_csv(RESULTS_DIR / 'feature_importance.csv')
print(f"Loaded {len(importance_df)} feature importance scores")

# ─────────────────────────────────────────────────────────────────────────────
# Test different feature selection strategies
# ─────────────────────────────────────────────────────────────────────────────

def compute_mds_stress(feat_matrix):
    scaler = StandardScaler()
    feat_norm = scaler.fit_transform(feat_matrix)
    dist_matrix = squareform(pdist(feat_norm, metric='euclidean'))
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds.fit(dist_matrix)
    return float(mds.stress_), mds

# Strategy 1: All 36 features (baseline)
feat_mask = np.ones(36, dtype=bool)
stress_all, mds_all = compute_mds_stress(feature_matrix_all)

# Strategy 2: Only positive importance features (top 3)
top_3_features = importance_df[importance_df['importance'] > 0]['feature'].values
feat_mask_top3 = np.array([f in top_3_features for f in all_feat_names])
feat_matrix_top3 = feature_matrix_all[:, feat_mask_top3]
stress_top3, mds_top3 = compute_mds_stress(feat_matrix_top3)

# Strategy 3: Top 10 features (mix positive + least negative)
top_10_features = importance_df.head(10)['feature'].values
feat_mask_top10 = np.array([f in top_10_features for f in all_feat_names])
feat_matrix_top10 = feature_matrix_all[:, feat_mask_top10]
stress_top10, mds_top10 = compute_mds_stress(feat_matrix_top10)

# Strategy 4: Top 15 (original approach)
top_15_features = importance_df.head(15)['feature'].values
feat_mask_top15 = np.array([f in top_15_features for f in all_feat_names])
feat_matrix_top15 = feature_matrix_all[:, feat_mask_top15]
stress_top15, mds_top15 = compute_mds_stress(feat_matrix_top15)

# Strategy 5: Top 20
top_20_features = importance_df.head(20)['feature'].values
feat_mask_top20 = np.array([f in top_20_features for f in all_feat_names])
feat_matrix_top20 = feature_matrix_all[:, feat_mask_top20]
stress_top20, mds_top20 = compute_mds_stress(feat_matrix_top20)

# Strategy 6: Only positive importance (all 3)
stress_pos3 = stress_top3
mds_pos3 = mds_top3
feat_pos3 = feat_matrix_top3

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

results = [
    {'strategy': 'All 36 features', 'n_features': 36, 'stress': stress_all, 'improvement': 0.0},
    {'strategy': 'Top 3 (pos importance)', 'n_features': 3, 'stress': stress_top3, 'improvement': stress_all - stress_top3},
    {'strategy': 'Top 10', 'n_features': 10, 'stress': stress_top10, 'improvement': stress_all - stress_top10},
    {'strategy': 'Top 15', 'n_features': 15, 'stress': stress_top15, 'improvement': stress_all - stress_top15},
    {'strategy': 'Top 20', 'n_features': 20, 'stress': stress_top20, 'improvement': stress_all - stress_top20},
]

results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("MDS STRESS ACROSS FEATURE SELECTION STRATEGIES")
print("=" * 100)
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Visualization: Stress comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Absolute stress
colors = ['red' if s == min(results_df['stress']) else 'steelblue' for s in results_df['stress']]
ax1.bar(range(len(results_df)), results_df['stress'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels([f"{r['strategy'].split('(')[0].strip()}\n({r['n_features']} features)"
                      for _, r in results_df.iterrows()], fontsize=9, rotation=0)
ax1.set_ylabel('MDS Stress', fontsize=11)
ax1.set_title('MDS Stress by Feature Selection Strategy', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, stress in enumerate(results_df['stress']):
    ax1.text(i, stress + 5, f'{stress:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right: Improvement over baseline
ax2.bar(range(len(results_df)), results_df['improvement'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels([f"{r['strategy'].split('(')[0].strip()}\n({r['n_features']} features)"
                      for _, r in results_df.iterrows()], fontsize=9, rotation=0)
ax2.set_ylabel('Stress Improvement over All-36', fontsize=11)
ax2.set_title('Stress Improvement vs Baseline (higher = better)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for i, imp in enumerate(results_df['improvement']):
    ax2.text(i, imp + (5 if imp >= 0 else -10), f'{imp:.1f}', ha='center',
            va='bottom' if imp >= 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
comp_path = RESULTS_DIR / 'mds_stress_comparison.png'
plt.savefig(comp_path, dpi=150, bbox_inches='tight')
print(f"\n  Saved: {comp_path}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Save summary
# ─────────────────────────────────────────────────────────────────────────────

summary_path = RESULTS_DIR / 'mds_stress_summary.csv'
results_df.to_csv(summary_path, index=False)
print(f"  Saved: {summary_path}")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)
best_idx = results_df['stress'].idxmin()
best = results_df.iloc[best_idx]
print(f"Best MDS stress: {best['stress']:.4f} using {best['n_features']} features ({best['strategy']})")
print(f"Top 3 positive-importance features:")
for i, row in importance_df[importance_df['importance'] > 0].iterrows():
    print(f"  - {row['feature']}: +{row['importance']:.4f}")
print("=" * 100)
