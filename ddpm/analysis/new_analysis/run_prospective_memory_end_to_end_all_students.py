#!/usr/bin/env python3
"""Run the prospective-memory end-to-end visualisation pipeline for many students.

By default this discovers student runs under `results_link_sampler` that match the
recovery-ablation naming pattern and invokes
`run_prospective_memory_end_to_end.py` for each run.

Outputs are written to one subdirectory per student run, preserving the run name.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_STUDENT_GLOBS = [
    "index_cued_first_diffusion_0.3_swap_recovery_ablation_*/",
    "index_cued_first_diffusion_0.3_swap_recovery_ablation_no_ablation_0/",
]

EXCLUDED_RUN_NAMES = {
    "index_cued_first_diffusion_0.3_swap_recovery_ablation_idk_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prospective-memory plots for all students")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results_link_sampler",
        help="Root directory containing student run folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "ddpm/analysis/new_analysis/results/prospective_memory_all_students",
        help="Directory where per-student outputs will be written",
    )
    parser.add_argument(
        "--student-glob",
        action="append",
        default=None,
        help="Optional glob pattern relative to results-root. Can be repeated.",
    )
    parser.add_argument("--angle-step", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--variance-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them")
    return parser.parse_args()


def discover_runs(results_root: Path, student_globs: list[str] | None) -> list[Path]:
    globs = student_globs or DEFAULT_STUDENT_GLOBS
    run_dirs: list[Path] = []
    seen: set[Path] = set()

    for pattern in globs:
        for run_dir in sorted(results_root.glob(pattern)):
            if not run_dir.is_dir():
                continue
            if run_dir in seen:
                continue
            if run_dir.name in EXCLUDED_RUN_NAMES:
                continue
            args_path = run_dir / "args.yaml"
            checkpoint_path = run_dir / "state.mdl"
            if not args_path.exists() or not checkpoint_path.exists():
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)

    return run_dirs


def main() -> None:
    args = parse_args()
    run_dirs = discover_runs(args.results_root, args.student_glob)

    if not run_dirs:
        raise FileNotFoundError(
            f"No student runs found under {args.results_root} using the configured glob pattern(s)."
        )

    print(f"Found {len(run_dirs)} student run(s):")
    for run_dir in run_dirs:
        print(f"- {run_dir.name}")

    if args.dry_run:
        return

    wrapper = REPO_ROOT / "ddpm/analysis/new_analysis/run_prospective_memory_end_to_end.py"
    args.output_root.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        output_dir = args.output_root / run_dir.name
        cmd = [
            sys.executable,
            str(wrapper),
            "--args-path",
            str(run_dir / "args.yaml"),
            "--checkpoint-path",
            str(run_dir / "state.mdl"),
            "--output-dir",
            str(output_dir),
            "--angle-step",
            str(args.angle_step),
            "--batch-size",
            str(args.batch_size),
            "--num-samples",
            str(args.num_samples),
            "--device",
            args.device,
            "--n-bins",
            str(args.n_bins),
            "--variance-threshold",
            str(args.variance_threshold),
            "--seed",
            str(args.seed),
        ]
        if args.force:
            cmd.append("--force")

        print("\n" + "=" * 80)
        print(f"Running: {run_dir.name}")
        print(f"Output:  {output_dir}")
        print("=" * 80)
        subprocess.run(cmd, check=True)

    print("\nAll student visualisations complete.")


if __name__ == "__main__":
    main()