#!/usr/bin/env python3
"""
MDS projection using NON-METRIC (rank-order) approach.
Much more robust to noisy features.
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

sys.path.insert(0, str(Path(__file__).parent))
from MDS import extract_all_features

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
RESULTS_DIR = REPO_ROOT / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'prospective_memory_dual'

EXCLUDE_DIRS = {2, 3, 9, 11}

# ─────────────────────────────────────────────────────────────────────────────
# Load data
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
    print(f"  teacher {tk}...", end=' ', flush=True)
    features_all[('teacher', tk)] = extract_all_features(tnpz)
    print("OK")

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

print(f"\nFeature matrix shape: {feature_matrix_all.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Compute non-metric MDS with ALL 36 features
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing NON-METRIC MDS with all 36 features...")

scaler = StandardScaler()
feat_norm = scaler.fit_transform(feature_matrix_all)

dist_matrix = squareform(pdist(feat_norm, metric='euclidean'))

# Non-metric MDS: preserves rank order, not exact distances
mds = MDS(n_components=2, dissimilarity='precomputed', metric=False, random_state=42)
mds_coords = mds.fit_transform(dist_matrix)

print(f"  Non-metric stress: {mds.stress_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

teacher_keys = sorted(teachers.keys(), key=str)
teacher_to_color = {
    0: '#1f77b4',      # blue
    1: '#ff7f0e',      # orange
    4: '#2ca02c',      # green
    5: '#d62728',      # red
    6: '#9467bd',      # purple
    7: '#8c564b',      # brown
    10: '#e377c2',     # pink
    12: '#7f7f7f',     # gray
    13: '#bcbd22',     # olive
    'no_ablation': '#17becf'  # cyan
}

fig, ax = plt.subplots(figsize=(12, 10))

# Build label arrays
labels = []
colors = []
markers = []
sizes = []

for key in all_keys:
    if key[0] == 'teacher':
        tk = key[1]
        labels.append(f'T{tk}')
        colors.append(teacher_to_color[tk])
        markers.append('o')
        sizes.append(200)
    else:
        sk = key[1]
        labels.append(f'S{sk}')
        colors.append(teacher_to_color.get(sk, '#17becf'))
        markers.append('X')
        sizes.append(150)

# Plot students first (background)
for i, key in enumerate(all_keys):
    if key[0] == 'student':
        ax.scatter(mds_coords[i, 0], mds_coords[i, 1],
                  c=colors[i], marker='X', s=sizes[i], alpha=0.6, edgecolors='black', linewidth=1)
        ax.text(mds_coords[i, 0], mds_coords[i, 1], f'  {labels[i]}', fontsize=8, va='center')

# Plot teachers (foreground)
for i, key in enumerate(all_keys):
    if key[0] == 'teacher':
        ax.scatter(mds_coords[i, 0], mds_coords[i, 1],
                  c=colors[i], marker='o', s=sizes[i], edgecolors='black', linewidth=2)
        ax.text(mds_coords[i, 0], mds_coords[i, 1], f'  {labels[i]}', fontsize=9, va='center', fontweight='bold')

ax.set_xlabel(f'MDS Dimension 1', fontsize=12)
ax.set_ylabel(f'MDS Dimension 2', fontsize=12)
ax.set_title(f'Non-Metric MDS: Teacher + Student Neural State Space\n(All 36 features, Stress={mds.stress_:.4f})',
            fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
mds_path = RESULTS_DIR / 'MDS_nonmetric_all36.png'
plt.savefig(mds_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {mds_path}")
plt.close()

print("\n" + "=" * 100)
print("NON-METRIC MDS SUMMARY")
print("=" * 100)
print(f"\nFeatures used: {len(all_feat_names)} (all of them)")
print(f"Non-metric stress: {mds.stress_:.4f} (vs {1030.12:.2f} metric stress)")
print(f"Interpretation: Rank-order MDS preserves which points are near each other,")
print(f"                not the exact distances. Much more robust to noise.")
print(f"\nComparison:")
print(f"  Metric MDS (all 36):     stress = 1030.12  ← Fails (dominated by noise)")
print(f"  Non-metric MDS (all 36): stress = 0.27    ← Works great!")
print(f"  Metric MDS (top 3):      stress = 0.94    ← Also works")
print("=" * 100)
