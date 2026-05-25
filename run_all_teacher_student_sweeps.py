"""
run_all_teacher_student_sweeps.py

For each teacher (healthy + all 14 ablation directions), find all corresponding
student runs and execute a compare_model_responses.py sweep.
Skips any teacher with no students. Runs sequentially.
"""

import subprocess
import sys
from pathlib import Path

BASE = Path("results_link_sampler")
TEACHER_RUN = BASE / "index_cued_first_diffusion_0.3_swap_7"
PREFIX = "index_cued_first_diffusion_0.3_swap_recovery_ablation_"
OUT_ROOT = Path("results/all_teacher_student_sweeps")
SCRIPT = Path("compare_model_responses.py")
MAX_SEED = 20  # scan up to this many seeds per direction


def find_students(direction_key: str) -> list[Path]:
    """Return sorted list of existing student directories for this direction."""
    students = []
    for seed in range(MAX_SEED):
        p = BASE / f"{PREFIX}{direction_key}_{seed}"
        if p.is_dir():
            students.append(p)
    return students


def run_sweep(direction_key: str, ablation_idx, students: list[Path]):
    label_teacher = f"Ablated_{direction_key}" if ablation_idx is not None else "Healthy"
    labels = [label_teacher] + [f"Student_{i}" for i in range(len(students))]

    run_paths = [str(TEACHER_RUN)] + [str(s) for s in students]
    abl_dirs = [str(ablation_idx) if ablation_idx is not None else "null"] + ["null"] * len(students)

    out_dir = OUT_ROOT / f"ablation_{direction_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cmd = [
        sys.executable, str(SCRIPT),
        "--run_paths", *run_paths,
        "--labels", *labels,
        "--ablation_directions", *abl_dirs,
        "--reference_idx", "0",
        "--student_vs_student",
        "--sweep",
        "--num_samples", "512",
        "--plot_subset", "6",
        "--out_dir", str(out_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Teacher: {label_teacher}  |  Students: {len(students)}")
    print(f"Output:  {out_dir}")
    print(f"{'='*60}")

    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode} — check {log_path}")
    else:
        # Print the summary lines from the log
        lines = log_path.read_text().splitlines()
        in_summary = False
        for line in lines:
            if line.startswith("Mean sliced-Wasserstein") or line.startswith("Reference self-SW") or line.startswith("Normalised SW"):
                in_summary = True
            if in_summary:
                print(f"  {line}")
            if in_summary and line.strip() == "" and "Normalised" not in line:
                break


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build list of (direction_key, ablation_idx) to check
    # no_ablation → healthy teacher, ablation_idx=None
    # 0..13       → ablated teacher, ablation_idx=int
    directions = [("no_ablation", None)] + [(str(d), d) for d in range(14)]

    skipped = []
    ran = []

    for direction_key, ablation_idx in directions:
        students = find_students(direction_key)
        if not students:
            skipped.append(direction_key)
            print(f"Skipping direction '{direction_key}' — no students found.")
            continue
        ran.append(direction_key)
        run_sweep(direction_key, ablation_idx, students)

    print(f"\n{'='*60}")
    print(f"Done. Ran: {ran}")
    print(f"Skipped (no students): {skipped}")


if __name__ == "__main__":
    main()
