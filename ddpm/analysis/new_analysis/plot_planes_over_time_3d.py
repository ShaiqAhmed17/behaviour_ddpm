"""Plot all fitted target/distractor planes over time in a single 3D figure.

Usage:
  python plot_planes_over_time_3d.py --repo-root /path/to/repo --run-path results_link_sampler/... 
  or
  python plot_planes_over_time_3d.py --master-summary /path/to/master_summary.json

The script loads `single_cue_target_distractor_planes_master_summary.json` and
draws semi-transparent planes and normals for each timepoint, colored by time.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="3D overlay of fitted planes across timepoints")
    p.add_argument("--master-summary", type=Path, default=None, help="Path to master summary JSON")
    p.add_argument("--repo-root", type=Path, default=Path("/scratch3/shaiq_home/repos/behaviour_ddpm"))
    p.add_argument("--run-path", type=Path, default=Path("results_link_sampler/index_cued_first_diffusion_0.3_swap_7"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--scale", type=float, default=1.0, help="Plane mesh scale multiplier")
    return p.parse_args()


def resolve_master_summary(repo_root, run_path, master_summary):
    if master_summary is not None:
        return master_summary
    run_dir = (repo_root / run_path).resolve()
    out = (
        repo_root
        / "ddpm"
        / "analysis"
        / "new_analysis"
        / "results"
        / run_dir.name
        / "single_cue_target_distractor_planes"
        / "single_cue_target_distractor_planes_master_summary.json"
    )
    return out


def create_plane_mesh(normal, center, size=1.0, n=10):
    # Create an oriented mesh in 3D for the plane
    normal = np.asarray(normal)
    if abs(normal[0]) < 0.9:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([0.0, 1.0, 0.0])
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    u = np.linspace(-size, size, n)
    v = np.linspace(-size, size, n)
    U, V = np.meshgrid(u, v)
    X = center[0] + U * v1[0] + V * v2[0]
    Y = center[1] + U * v1[1] + V * v2[1]
    Z = center[2] + U * v1[2] + V * v2[2]
    return X, Y, Z


def main():
    args = parse_args()
    master_summary_path = resolve_master_summary(args.repo_root, args.run_path, args.master_summary)
    if not master_summary_path.exists():
        raise FileNotFoundError(f"Master summary not found: {master_summary_path}")

    with open(master_summary_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    per = master.get("per_timepoint_results") or master.get("per_prep_results")
    if per is None:
        raise RuntimeError("No per-timepoint results found in master summary")

    keys = list(per.keys())
    # sort keys by natural order prep_0, prep_1, ..., diff_0, diff_1
    def key_sort(k):
        if isinstance(k, int):
            return (0, k)
        s = str(k)
        if s.startswith("prep_"):
            return (0, int(s.split("_")[1]))
        if s.startswith("diff_"):
            return (1, int(s.split("_")[1]))
        return (2, s)

    keys = sorted(keys, key=key_sort)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    n_keys = len(keys)
    cmap = plt.cm.viridis

    all_centers = []
    for idx, k in enumerate(keys):
        entry = per[k]
        # entry contains target_center/normal and distractor_center/normal
        t_center = np.array(entry["target_center"])
        t_normal = np.array(entry["target_normal"])
        d_center = np.array(entry["distractor_center"])
        d_normal = np.array(entry["distractor_normal"])

        color = cmap(float(idx) / max(1, n_keys - 1))
        size = args.scale * 1.0
        TX, TY, TZ = create_plane_mesh(t_normal, t_center, size=size, n=8)
        DX, DY, DZ = create_plane_mesh(d_normal, d_center, size=size, n=8)

        ax.plot_surface(TX, TY, TZ, color=color, alpha=0.12, linewidth=0)
        ax.plot_surface(DX, DY, DZ, color=color, alpha=0.12, linewidth=0)

        # normals as arrows
        ax.quiver(
            t_center[0], t_center[1], t_center[2],
            t_normal[0] * size * 0.6, t_normal[1] * size * 0.6, t_normal[2] * size * 0.6,
            color=color, linewidth=1.5
        )
        ax.quiver(
            d_center[0], d_center[1], d_center[2],
            d_normal[0] * size * 0.6, d_normal[1] * size * 0.6, d_normal[2] * size * 0.6,
            color=color, linewidth=1.0
        )

        all_centers.append(t_center)
        all_centers.append(d_center)

    all_centers = np.array(all_centers)
    mins = all_centers.min(axis=0)
    maxs = all_centers.max(axis=0)
    span = maxs - mins
    pad = 0.2 * span
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Target (and Distractor) planes over time')

    out_path = args.out or master_summary_path.parent / "single_3d_planes_over_time.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved 3D overlay to: {out_path}")


if __name__ == '__main__':
    main()
