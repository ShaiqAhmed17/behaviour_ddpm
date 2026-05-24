#!/usr/bin/env python3
"""CLI wrapper for prospective memory global 3D PCA trajectory plots.

This script lets you run plotting from terminal without notebooks.
It supports three input styles in a single .npz file:
1) precomputed points + labels
2) preparatory state sequences + metadata
3) diffusion state sequences + metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import matplotlib

# Headless-safe backend for terminal/cluster usage.
matplotlib.use("Agg")

import numpy as np
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddpm.analysis.new_analysis.prospective_memory_plotting import (
    make_full_step_diffusion_dataset,
    make_full_step_trajectory_dataset,
    plot_global_3d_trajectories,
)


def _parse_prep_indices(raw: str | None, available: Iterable[int]) -> list[int]:
    if raw is None:
        return sorted(int(x) for x in available)
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    out = [int(v) for v in vals]
    return out


def _resolve_mode(npz: np.lib.npyio.NpzFile, mode: str, keys: dict[str, str]) -> str:
    points_key = keys["points"]
    labels_key = keys["labels"]
    prep_key = keys["prep"]
    diffusion_key = keys["diffusion"]

    if mode != "auto":
        return mode

    if points_key in npz.files and labels_key in npz.files:
        return "precomputed"
    if prep_key in npz.files:
        return "prep"
    if diffusion_key in npz.files:
        return "diffusion"

    raise KeyError(
        "Could not infer mode from NPZ keys. Expected one of: "
        f"({points_key},{labels_key}) or {prep_key} or {diffusion_key}. "
        f"Available keys: {npz.files}"
    )


def _load_prep_dataset(
    npz: np.lib.npyio.NpzFile,
    metadata_key: str,
    prep_key: str,
    prep_indices: str | None,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    metadata = np.asarray(npz[metadata_key])
    prep_obj = npz[prep_key]
    if not (prep_obj.dtype == object and prep_obj.shape == ()):  # noqa: E721
        raise ValueError(
            f"Expected object scalar dict at key '{prep_key}', got shape={prep_obj.shape}, dtype={prep_obj.dtype}."
        )

    states_seq_by_prep = prep_obj.item()
    indices = _parse_prep_indices(prep_indices, states_seq_by_prep.keys())

    points, labels, prep_step_counts = make_full_step_trajectory_dataset(
        states_seq_by_prep=states_seq_by_prep,
        metadata=metadata,
        prep_indices=indices,
        n_bins=n_bins,
    )
    print(f"Loaded prep dataset with prep indices {indices} and step counts {prep_step_counts}")
    return points, labels, "prep"


def _load_diffusion_dataset(
    npz: np.lib.npyio.NpzFile,
    metadata_key: str,
    diffusion_key: str,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    metadata = np.asarray(npz[metadata_key])
    states_seq_by_diffusion = np.asarray(npz[diffusion_key])

    points, labels, t_diff = make_full_step_diffusion_dataset(
        states_seq_by_diffusion=states_seq_by_diffusion,
        metadata=metadata,
        n_bins=n_bins,
    )
    print(f"Loaded diffusion dataset with T_diff={t_diff}")
    return points, labels, "diffusion"


def _load_precomputed_dataset(
    npz: np.lib.npyio.NpzFile,
    points_key: str,
    labels_key: str,
    label_schema: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    points = np.asarray(npz[points_key])
    labels = np.asarray(npz[labels_key])

    if label_schema == "auto":
        if labels.ndim != 2:
            raise ValueError(f"Labels must be rank-2, got shape={labels.shape}")
        if labels.shape[1] in (6, 7):
            schema = "prep"
        elif labels.shape[1] == 4:
            schema = "diffusion"
        else:
            raise ValueError(
                "Cannot infer label schema from labels shape. "
                f"Expected second dim 6 or 4, got {labels.shape[1]}."
            )
    else:
        schema = label_schema

    print(f"Loaded precomputed dataset with schema='{schema}'")
    return points, labels, schema


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone wrapper for prospective memory 3D PCA plotting")

    p.add_argument("--input", type=Path, required=True, help="Path to input .npz")
    p.add_argument("--output", type=Path, required=True, help="Path to output figure (.png)")
    p.add_argument("--title", type=str, default="Prospective memory global PCA trajectories")

    p.add_argument(
        "--mode",
        choices=["auto", "prep", "diffusion", "precomputed"],
        default="auto",
        help="How to interpret NPZ content",
    )
    p.add_argument("--n-bins", type=int, default=12, help="Color bins for cued-angle grouping")
    p.add_argument("--prep-indices", type=str, default=None, help="Comma-separated prep indices, e.g. 0,1,2,3")

    p.add_argument("--metadata-key", type=str, default="metadata")
    p.add_argument("--prep-key", type=str, default="states_seq_by_prep")
    p.add_argument("--diffusion-key", type=str, default="states_seq_by_diffusion")
    p.add_argument("--points-key", type=str, default="points")
    p.add_argument("--labels-key", type=str, default="labels")

    p.add_argument(
        "--label-schema",
        choices=["auto", "prep", "diffusion"],
        default="auto",
        help="Only used in precomputed mode",
    )

    p.add_argument("--n-components", type=int, default=3, help="PCA components to fit (must be >= 3)")
    p.add_argument(
        "--variance-threshold",
        type=float,
        default=0.9,
        help="Threshold in [0,1] for minimal-PC summary",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")
    if not (0.0 < args.variance_threshold <= 1.0):
        raise ValueError("--variance-threshold must be in (0, 1]")

    keymap = {
        "metadata": args.metadata_key,
        "prep": args.prep_key,
        "diffusion": args.diffusion_key,
        "points": args.points_key,
        "labels": args.labels_key,
    }

    with np.load(args.input, allow_pickle=True) as data:
        mode = _resolve_mode(data, args.mode, keymap)
        print(f"Resolved mode: {mode}")

        if mode == "prep":
            points, labels, schema = _load_prep_dataset(
                data,
                metadata_key=args.metadata_key,
                prep_key=args.prep_key,
                prep_indices=args.prep_indices,
                n_bins=args.n_bins,
            )
        elif mode == "diffusion":
            points, labels, schema = _load_diffusion_dataset(
                data,
                metadata_key=args.metadata_key,
                diffusion_key=args.diffusion_key,
                n_bins=args.n_bins,
            )
        else:
            points, labels, schema = _load_precomputed_dataset(
                data,
                points_key=args.points_key,
                labels_key=args.labels_key,
                label_schema=args.label_schema,
            )

    if points.ndim != 2:
        raise ValueError(f"points must be rank-2 [M,D], got shape={points.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be rank-2 [M,K], got shape={labels.shape}")
    if points.shape[0] == 0:
        raise ValueError("No points to plot after preprocessing")
    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"points and labels row mismatch: {points.shape[0]} vs {labels.shape[0]}")

    n_comp = min(max(int(args.n_components), 3), points.shape[0], points.shape[1])
    if n_comp < 3:
        raise ValueError(
            "Need at least 3 PCA components for 3D plot. "
            f"Available samples/features imply n_components={n_comp}."
        )

    pca = PCA(n_components=n_comp)
    pca_coords = pca.fit_transform(points)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k_min = int(np.searchsorted(cumulative, args.variance_threshold) + 1)
    if k_min > len(cumulative):
        k_min = len(cumulative)

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("PCA cumulative variance:", cumulative)
    print(
        f"Minimal PCs for threshold {args.variance_threshold:.1%}: {k_min} "
        f"(cumulative={cumulative[k_min - 1]:.3f})"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_global_3d_trajectories(
        pca_coords=pca_coords,
        labels=labels,
        pca=pca,
        n_bins=args.n_bins,
        out_path=args.output,
        title=args.title,
        label_schema=schema,
    )


if __name__ == "__main__":
    main()
