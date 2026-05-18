#!/usr/bin/env python3
"""Compare healthy vs ablated teacher sweeps using sliced-Wasserstein distance.

Saves:
- wasserstein_ranking.csv
- wasserstein_per_trial.csv
- wasserstein_bar.png
- wasserstein_heatmap.png
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance


def load_sweep(path):
    return torch.load(path, weights_only=False)


def ensure_consistent(healthy, ablated_list):
    h_meta = healthy["meta"]
    n_trials = len(h_meta["trial_ids"]) if "trial_ids" in h_meta else healthy["states"].shape[0]
    n_samples = h_meta.get("n_samples", healthy["states"].shape[1])
    for ab in ablated_list:
        ameta = ab["meta"]
        if ameta.get("n_samples", ab["states"].shape[1]) != n_samples:
            raise ValueError("Mismatch in n_samples between healthy and ablation %s" % str(ab.get("direc_idx")))
        if len(ameta.get("trial_ids", list(range(n_trials)))) != n_trials:
            raise ValueError("Mismatch in n_trials between healthy and ablation %s" % str(ab.get("direc_idx")))
    return n_trials, n_samples


def _random_unit_directions(dim, n_projections, rng):
    vecs = rng.normal(size=(n_projections, dim)).astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def sliced_wasserstein_distance(X, Y, n_projections=128, seed=42):
    """Compute sliced-Wasserstein distance between two point clouds.

    X, Y: arrays of shape [n_samples, D]
    Returns scalar average Wasserstein across random 1D projections.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays [n_samples, D]")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have same feature dimension")

    dim = X.shape[1]
    rng = np.random.default_rng(seed)
    dirs = _random_unit_directions(dim, n_projections, rng)

    dists = []
    for v in dirs:
        x_proj = X @ v
        y_proj = Y @ v
        dists.append(wasserstein_distance(x_proj, y_proj))
    return float(np.mean(dists))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-projections", type=int, default=128,
                        help="Number of random projections for sliced-Wasserstein")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for projection directions")
    args = parser.parse_args()

    sweeps_dir = Path(args.sweeps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    healthy_path = sweeps_dir / "healthy.pt"
    if not healthy_path.exists():
        raise FileNotFoundError("healthy.pt not found in %s" % sweeps_dir)
    healthy = load_sweep(healthy_path)

    # find ablated files
    ablated_files = sorted(sweeps_dir.glob("ablated_dir_*.pt"))
    if len(ablated_files) == 0:
        raise FileNotFoundError("No ablated_dir_*.pt files found in %s" % sweeps_dir)

    ablated = [load_sweep(p) for p in ablated_files]

    n_trials, n_samples = ensure_consistent(healthy, ablated)

    H = np.asarray(healthy["states"])  # shape [n_trials, n_samples, D]
    if H.ndim != 3:
        raise ValueError(f"healthy['states'] must be 3D [n_trials, n_samples, D], got {H.shape}")

    per_trial_table = []
    ranking_rows = []

    for ab in ablated:
        di = ab.get("direc_idx")
        A = np.asarray(ab["states"])
        if A.shape != H.shape:
            raise ValueError(f"Shape mismatch for direction {di}: healthy {H.shape} vs ablated {A.shape}")

        per_trial = []
        for t in range(n_trials):
            wd = sliced_wasserstein_distance(
                H[t],
                A[t],
                n_projections=args.n_projections,
                seed=args.seed + int(di) * 1000 + t,
            )
            per_trial.append(wd)
        per_trial_table.append(per_trial)
        arr = np.array(per_trial)
        ranking_rows.append({
            "direc_idx": int(di),
            "metric": "sliced_wasserstein",
            "n_projections": int(args.n_projections),
            "mean_wasserstein": float(arr.mean()),
            "median_wasserstein": float(np.median(arr)),
            "max_wasserstein": float(arr.max()),
            "std_wasserstein": float(arr.std()),
        })

    # Save per-trial CSV
    per_trial_np = np.array(per_trial_table)  # shape [n_dirs, n_trials]
    df_per_trial = pd.DataFrame(per_trial_np, index=[int(x.get("direc_idx")) for x in ablated],
                                columns=[f"trial_{i}" for i in range(n_trials)])
    df_per_trial.index.name = "direc_idx"
    df_per_trial.to_csv(out_dir / "wasserstein_per_trial.csv")

    # Save ranking
    df_rank = pd.DataFrame(ranking_rows).sort_values("mean_wasserstein", ascending=False)
    df_rank.to_csv(out_dir / "wasserstein_ranking.csv", index=False)

    # Bar plot
    plt.figure(figsize=(8, 4))
    sns.barplot(x="direc_idx", y="mean_wasserstein", data=df_rank)
    plt.title(f"Mean Sliced-Wasserstein per direction (n_proj={args.n_projections})")
    plt.tight_layout()
    plt.savefig(out_dir / "wasserstein_bar.png", dpi=150)
    plt.close()

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_per_trial.astype(float), cmap="viridis", cbar_kws={"label": "Sliced-Wasserstein"})
    plt.xlabel("trial")
    plt.ylabel("direction_idx")
    plt.tight_layout()
    plt.savefig(out_dir / "wasserstein_heatmap.png", dpi=150)
    plt.close()

    # Print top-5
    top5 = df_rank.head(5)
    print("Top-5 directions by mean sliced-Wasserstein:")
    print(top5[["direc_idx", "mean_wasserstein"]].to_string(index=False))

    print("Saved sliced-Wasserstein CSVs and plots to", out_dir)


if __name__ == "__main__":
    main()
