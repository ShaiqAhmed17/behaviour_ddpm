#!/usr/bin/env python3
"""Compute prep-trajectory geometry summaries per ablation direction.

Outputs:
- prep_geometry_summary.csv
- teacher_ablation_impact_ranked.csv (merged with wasserstein_ranking.csv if present)
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from scipy.stats import pearsonr


def load_sweep(path):
    return torch.load(path, weights_only=False)


def centroid_distance(H, A):
    # H, A: [n_trials, n_samples, D]
    cent_H = H.mean(axis=1)  # [n_trials, D]
    cent_A = A.mean(axis=1)
    d = np.linalg.norm(cent_H - cent_A, axis=1)
    return d, float(d.mean()), float(d.std())


def plane_angle_deg(H, A, k=2):
    # compute k-dim principal angles between subspaces spanned by top-k PCs of H and A
    # flatten across trials & samples
    XH = H.reshape(-1, H.shape[2])
    XA = A.reshape(-1, A.shape[2])
    XHc = XH - XH.mean(axis=0, keepdims=True)
    XAc = XA - XA.mean(axis=0, keepdims=True)
    # get top-k orthonormal bases
    _, _, VtH = svd(XHc, full_matrices=False)
    _, _, VtA = svd(XAc, full_matrices=False)
    UH = VtH[:k].T  # D x k
    UA = VtA[:k].T
    M = UH.T @ UA
    s = svd(M, compute_uv=False)
    # clamp
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    # return mean principal angle in degrees
    return float(np.degrees(angles).mean())


def ring_radius_stats(H, A):
    # per-trial mean radius difference and mean radius for healthy
    cent_H = H.mean(axis=1, keepdims=True)
    R_H = np.linalg.norm(H - cent_H, axis=2)  # [n_trials, n_samples]
    cent_A = A.mean(axis=1, keepdims=True)
    R_A = np.linalg.norm(A - cent_A, axis=2)
    mean_RH = R_H.mean(axis=1)
    mean_RA = R_A.mean(axis=1)
    diff = mean_RA - mean_RH
    return float(mean_RH.mean()), float(diff.mean()), float(diff.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--merge-wasserstein", default=None,
                        help="Optional path to wasserstein_ranking.csv. If omitted, tries <sweeps-dir>/wasserstein_ranking.csv")
    args = parser.parse_args()

    sweeps_dir = Path(args.sweeps_dir)
    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    healthy = load_sweep(sweeps_dir / "healthy.pt")
    ablated_files = sorted(sweeps_dir.glob("ablated_dir_*.pt"))
    ablated = [load_sweep(p) for p in ablated_files]

    if len(ablated) != 14:
        raise ValueError(f"Expected exactly 14 ablated sweeps, found {len(ablated)} in {sweeps_dir}")

    H = np.array(healthy["states"])
    rows = []
    for ab in ablated:
        di = int(ab.get("direc_idx"))
        key = ab.get("direc_key")
        A = np.array(ab["states"])
        # validation shapes
        if H.shape != A.shape:
            raise ValueError(f"Shape mismatch healthy vs ablated {di}: {H.shape} vs {A.shape}")

        per_trial_centd, mean_centd, std_centd = centroid_distance(H, A)
        plane_ang = plane_angle_deg(H, A, k=2)
        mean_RH, mean_R_diff, std_R_diff = ring_radius_stats(H, A)

        rows.append({
            "direc_idx": di,
            "direc_key": key,
            "centroid_mean_dist": mean_centd,
            "centroid_std_dist": std_centd,
            "plane_angle_deg": plane_ang,
            "mean_ring_radius_healthy": mean_RH,
            "mean_ring_radius_diff": mean_R_diff,
            "mean_ring_radius_diff_std": std_R_diff,
        })

    df = pd.DataFrame(sorted(rows, key=lambda x: x["direc_idx"]))
    df.to_csv(out_csv, index=False)
    print("Wrote prep geometry summary:", out_csv)

    # Merge with wasserstein ranking (explicit path or default in sweeps_dir)
    merge_path = args.merge_wasserstein
    if merge_path is None:
        candidate = sweeps_dir / "wasserstein_ranking.csv"
        if candidate.exists():
            merge_path = str(candidate)

    if merge_path is None:
        print("No wasserstein ranking found. Skipping merged ranking output.")
        return

    wr = pd.read_csv(merge_path)
    if len(wr) != 14:
        raise ValueError(f"Expected exactly 14 rows in wasserstein ranking, found {len(wr)} in {merge_path}")

    merged = wr.merge(df, on="direc_idx", how="inner")
    if len(merged) != 14:
        raise ValueError(f"Merged ranking should have 14 rows, found {len(merged)}")

    merged = merged.sort_values("mean_wasserstein", ascending=False)
    out_rank = out_dir / "teacher_ablation_impact_ranked.csv"
    merged.to_csv(out_rank, index=False)
    print("Wrote merged ranking:", out_rank)

    # Correlation-based interpretation between impact and geometry metrics
    impact = merged["mean_wasserstein"].to_numpy(dtype=float)
    geom_cols = [
        "centroid_mean_dist",
        "plane_angle_deg",
        "mean_ring_radius_diff",
        "mean_ring_radius_diff_std",
    ]
    print("\nCorrelation with mean Wasserstein impact:")
    for c in geom_cols:
        vals = merged[c].to_numpy(dtype=float)
        r, p = pearsonr(impact, vals)
        print(f"  {c}: r={r:.4f}, p={p:.3g}")

    top5 = merged.head(5)[["direc_idx", "mean_wasserstein"]]
    print("\nTop-5 highest-impact directions:")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
