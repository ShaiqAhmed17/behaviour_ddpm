#!/usr/bin/env python3
"""
Rank all features by their contribution to MDS using leave-one-out analysis.
Remove each feature one at a time, recompute MDS, measure stress increase.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import re
import sys

# Import the feature extraction code from MDS.py
sys.path.insert(0, str(Path(__file__).parent))
from MDS import extract_all_features
from ddpm.utils.vis.style import set_publication_style

set_publication_style()

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_DIR = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'

EXCLUDE_DIRS = {2, 3, 9, 11}

# ─────────────────────────────────────────────────────────────────────────────
# Load NPZs
# ─────────────────────────────────────────────────────────────────────────────

print("Scanning NPZ files (excluding directions 2, 3, 9, 11)...")
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

print(f"  Teachers: {len(teachers)}")
print(f"  Students: {sum(len(v) for v in students.values())}")

# ─────────────────────────────────────────────────────────────────────────────
# Extract features
# ─────────────────────────────────────────────────────────────────────────────

print("\nExtracting features...")
features_all = {}

for tk, tnpz in sorted(teachers.items(), key=str):
    print(f"  teacher {tk}...", end=' ', flush=True)
    features_all[('teacher', tk)] = extract_all_features(tnpz)
    print("OK")

for sk, stu_list in sorted(students.items(), key=str):
    for sname, snpz in stu_list:
        features_all[('student', sk, sname)] = extract_all_features(snpz)

print(f"\nExtracted {len(features_all)} feature vectors")

# ─────────────────────────────────────────────────────────────────────────────
# Build feature matrix with ALL features (no selection)
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding feature matrix...")

# All feature names
all_feat_names = sorted(list(features_all[('teacher', list(teachers.keys())[0])].keys()))
print(f"  Total features: {len(all_feat_names)}")

# Build feature matrix for all points (teachers + students)
all_keys = (
    [(('teacher', tk)) for tk in sorted(teachers.keys(), key=str)] +
    [(('student', sk, sname)) for sk in sorted(students.keys(), key=str)
     for sname, _ in sorted(students[sk])]
)

feature_matrix = np.array([[features_all[key][fname] for fname in all_feat_names]
                           for key in all_keys])
print(f"  Feature matrix shape: {feature_matrix.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper function: compute MDS stress given a feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_mds_stress(feat_matrix):
    """Compute MDS stress for a feature matrix."""
    scaler = StandardScaler()
    feat_norm = scaler.fit_transform(feat_matrix)

    # Compute distance matrix
    dist_matrix = squareform(pdist(feat_norm, metric='euclidean'))

    # Fit MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds.fit(dist_matrix)

    return float(mds.stress_)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline: MDS stress with all features
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing baseline MDS stress (all features)...")
baseline_stress = compute_mds_stress(feature_matrix)
print(f"  Baseline stress: {baseline_stress:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-out: remove each feature and measure stress increase
# ─────────────────────────────────────────────────────────────────────────────

print("\nLeave-one-out analysis (this may take a minute)...")
importance_scores = {}

for i, fname in enumerate(all_feat_names):
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(all_feat_names)}...", end=' ', flush=True)

    # Remove feature i
    mask = np.ones(len(all_feat_names), dtype=bool)
    mask[i] = False
    feat_matrix_without = feature_matrix[:, mask]

    # Compute stress without this feature
    stress_without = compute_mds_stress(feat_matrix_without)

    # Importance = how much stress increases when feature is removed
    importance = stress_without - baseline_stress
    importance_scores[fname] = {
        'stress_without': stress_without,
        'importance': importance
    }

    if (i + 1) % 5 == 0:
        print(f"importance={importance:.4f}")

print("\n" + "=" * 100)
print("FEATURE IMPORTANCE RANKING (by stress increase when removed)")
print("=" * 100)

# Sort by importance (descending)
sorted_features = sorted(importance_scores.items(), key=lambda x: x[1]['importance'], reverse=True)

importance_df = pd.DataFrame([
    {
        'rank': i+1,
        'feature': fname,
        'importance': scores['importance'],
        'stress_without': scores['stress_without']
    }
    for i, (fname, scores) in enumerate(sorted_features)
])

print(importance_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

csv_path = RESULTS_DIR / 'feature_importance.csv'
importance_df.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Bar plot of top 20 features
# ─────────────────────────────────────────────────────────────────────────────

print("\nPlotting feature importance...")

fig, ax = plt.subplots(figsize=(12, 8))

top_k = 20
top_features = importance_df.head(top_k)

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values, fontsize=10)
ax.set_xlabel('Importance (stress increase when feature removed)', fontsize=11)
ax.set_title(f'Top {top_k} Most Important Features for MDS (Baseline stress: {baseline_stress:.2f})',
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['importance'] + 0.02, i, f"{row['importance']:.4f}",
            va='center', fontsize=9)

plt.tight_layout()
barplot_path = RESULTS_DIR / 'feature_importance_top20.png'
plt.savefig(barplot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {barplot_path}")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Visualization 2: Cumulative importance
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

cumsum_importance = importance_df['importance'].cumsum()
ax.plot(range(len(importance_df)), cumsum_importance.values, 'o-', linewidth=2, markersize=6)

# Mark 80%, 90%, 95% of total
total_importance = cumsum_importance.values[-1]
for pct in [80, 90, 95]:
    threshold = total_importance * pct / 100
    n_features = np.argmax(cumsum_importance.values >= threshold) + 1
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.3)
    ax.axvline(n_features - 1, color='red', linestyle='--', alpha=0.3)
    ax.text(n_features - 1, threshold, f'{pct}% @ {n_features} features',
            fontsize=9, ha='left', va='bottom')

ax.set_xlabel('Number of Features', fontsize=11)
ax.set_ylabel('Cumulative Importance', fontsize=11)
ax.set_title('Cumulative Feature Importance (how many features explain how much stress)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
cumplot_path = RESULTS_DIR / 'feature_importance_cumulative.png'
plt.savefig(cumplot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {cumplot_path}")
plt.close()

print("\n" + "=" * 100)
print(f"✓ Feature importance analysis complete!")
print(f"  Baseline stress: {baseline_stress:.4f}")
print(f"  Total features: {len(all_feat_names)}")
print("=" * 100)
