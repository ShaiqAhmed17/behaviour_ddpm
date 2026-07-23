#!/usr/bin/env python3
"""
Analyze and visualize top 15 features across teacher ablation directions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from MDS import extract_all_features

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

    # Single/multi-dir nullspace ablation: ...recovery_ablation_{key}_{seed}
    m_student = re.match(r'^index_cued_first_diffusion_0\.3_swap_recovery_ablation_(.+)_(\d+)$', d.name)
    if m_student:
        key = m_student.group(1)
        try:
            key = int(key)
        except ValueError:
            pass
        if key != 'idk' and key not in EXCLUDE_DIRS:
            students.setdefault(key, []).append((d.name, npz))
        continue

    # PCA-based ablation: ...recovery_pca_ablation_{dirs}_{seed}
    m_pca = re.match(r'^index_cued_first_diffusion_0\.3_swap_recovery_pca_ablation_(.+)_(\d+)$', d.name)
    if m_pca:
        key = f'pca_{m_pca.group(1)}'
        students.setdefault(key, []).append((d.name, npz))
        continue

    # No-ablation recovery: ...recovery_no_ablation_{seed}
    m_noabl = re.match(r'^index_cued_first_diffusion_0\.3_swap_recovery_no_ablation_(\d+)$', d.name)
    if m_noabl:
        key = 'no_ablation_v2'
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
# Feature selection
# ─────────────────────────────────────────────────────────────────────────────

print("\nFeature selection...")

# Use all features
all_feat_names = list(features_all[('teacher', list(teachers.keys())[0])].keys())
print(f"  Total features: {len(all_feat_names)}")

# Compute CV (std / |mean|) across teachers — ranks features by relative variability,
# factoring out differences in magnitude
teacher_features = np.array([[features_all[('teacher', tk)][fn] for fn in all_feat_names]
                             for tk in sorted(teachers.keys(), key=str)])
eps = 1e-8
feat_cv = teacher_features.std(axis=0) / (np.abs(teacher_features.mean(axis=0)) + eps)

top_k = 15
top_k_idx = np.argsort(feat_cv)[-top_k:][::-1]
top_k_names = [all_feat_names[i] for i in top_k_idx]

print(f"\nTop {top_k} features by CV (std/|mean|):")
for i, idx in enumerate(top_k_idx):
    print(f"  {i+1:2d}. {all_feat_names[idx]:40s} cv={feat_cv[idx]:.6e}")

# ─────────────────────────────────────────────────────────────────────────────
# Build feature matrix for teachers only
# ─────────────────────────────────────────────────────────────────────────────

teacher_keys = sorted(teachers.keys(), key=str)
teacher_labels = [str(k) for k in teacher_keys]

teacher_feature_matrix = np.array([[features_all[('teacher', tk)][fname] for fname in top_k_names]
                                   for tk in teacher_keys])

# ─────────────────────────────────────────────────────────────────────────────
# Create DataFrame for easy inspection
# ─────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(teacher_feature_matrix, columns=top_k_names, index=teacher_labels)

print("\n" + "=" * 100)
print("TEACHER FEATURE VALUES")
print("=" * 100)
print(df.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Heatmap of features
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Visualization 1: Heatmap of features
# ─────────────────────────────────────────────────────────────────────────────

from ddpm.utils.vis.style import set_publication_style, save_figure

set_publication_style()

print("\nPlotting heatmap...")

fig, ax = plt.subplots(figsize=(14, 8))

# Normalize each feature to [0, 1] for better heatmap visualization
teacher_feature_matrix_norm = (teacher_feature_matrix - teacher_feature_matrix.min(axis=0)) / \
                              (teacher_feature_matrix.max(axis=0) - teacher_feature_matrix.min(axis=0) + 1e-10)

im = ax.imshow(teacher_feature_matrix_norm.T, aspect='auto', cmap='RdYlGn', interpolation='nearest')

ax.set_xticks(range(len(teacher_labels)))
ax.set_xticklabels(teacher_labels, rotation=45, ha='right')
ax.set_yticks(range(len(top_k_names)))
ax.set_yticklabels([f.replace('_', '\n')[:30] for f in top_k_names], fontsize=8)

ax.set_xlabel('Teacher Ablation Direction', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_title('Top 15 Features: Normalized Heatmap Across Teachers', fontsize=13, fontweight='bold')

plt.colorbar(im, ax=ax, label='Normalized Value')
plt.tight_layout()

heatmap_path = RESULTS_DIR / 'top15_features_heatmap'
save_figure(fig, heatmap_path)
print(f"  Saved: {heatmap_path}.pdf / .png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Visualization 2: Line plots for each feature
# ─────────────────────────────────────────────────────────────────────────────

print("Plotting feature trajectories...")

fig, axes = plt.subplots(3, 5, figsize=(18, 12))
axes = axes.flatten()

for i, fname in enumerate(top_k_names):
    ax = axes[i]

    # Get values for this feature across teachers
    vals = [features_all[('teacher', tk)][fname] for tk in teacher_keys]

    # Plot
    ax.plot(range(len(teacher_labels)), vals, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xticks(range(len(teacher_labels)))
    ax.set_xticklabels(teacher_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title(fname.replace('_', ' ')[:40], fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('Top 15 Features: Values Across Teacher Ablation Directions', fontsize=14, fontweight='bold')
plt.tight_layout()

lineplot_path = RESULTS_DIR / 'top15_features_lineplots'
save_figure(fig, lineplot_path)
print(f"  Saved: {lineplot_path}.pdf / .png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────────────────────────────────────

csv_path = RESULTS_DIR / 'top15_features.csv'
df.to_csv(csv_path)
print(f"  Saved: {csv_path}")

print("\n" + "=" * 100)
print(f"✓ Feature analysis complete!")
print("=" * 100)
