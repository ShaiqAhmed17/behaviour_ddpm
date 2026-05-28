"""
run_all_teacher_student_sweeps.py

For each teacher seed and each ablation direction (healthy + 14 ablations),
find all corresponding student runs and execute a compare_model_responses.py sweep.
Each student is compared against the teacher it was trained from.

Student naming conventions:
  Legacy (teacher_id implicitly "7", existing runs):
    ...swap_recovery_ablation_{dir}_{seed}
    ...swap_recovery_no_ablation_{seed}
  New (explicit teacher_id):
    ...swap_recovery_teacher{id}_ablation_{dir}_{seed}
    ...swap_recovery_teacher{id}_no_ablation_{seed}
"""

import subprocess
import sys
from pathlib import Path

BASE = Path("results_link_sampler")
STUDENT_PREFIX = "index_cued_first_diffusion_0.3_swap_recovery"
OUT_ROOT = Path("results/all_teacher_student_sweeps")
SCRIPT = Path("compare_model_responses.py")
MAX_SEED = 20

TEACHER_SEEDS = ["0", "1", "3", "4", "7"]


def teacher_run_dir(teacher_id: str) -> Path:
    return BASE / f"index_cued_first_diffusion_0.3_swap_{teacher_id}"


def find_students(direction_key: str, teacher_id: str) -> list[Path]:
    """Return existing student directories for this (teacher, direction) pair."""
    students = []
    for seed in range(MAX_SEED):
        # New-style: teacher_id encoded in path
        if direction_key == "no_ablation":
            candidates = [
                BASE / f"{STUDENT_PREFIX}_teacher{teacher_id}_no_ablation_{seed}",
            ]
        else:
            candidates = [
                BASE / f"{STUDENT_PREFIX}_teacher{teacher_id}_ablation_{direction_key}_{seed}",
            ]

        # Legacy paths (no teacher tag) — only for the legacy default teacher "7"
        if teacher_id == "7":
            if direction_key == "no_ablation":
                candidates += [
                    BASE / f"{STUDENT_PREFIX}_ablation_no_ablation_{seed}",
                    BASE / f"{STUDENT_PREFIX}_no_ablation_{seed}",
                ]
            else:
                candidates.append(BASE / f"{STUDENT_PREFIX}_ablation_{direction_key}_{seed}")

        for p in candidates:
            if p.is_dir():
                students.append(p)
                break  # only one match per seed slot

    return students


def run_sweep(teacher_id: str, direction_key: str, ablation_idx, students: list[Path]):
    label_teacher = f"Ablated_{direction_key}" if ablation_idx is not None else "Healthy"
    labels = [label_teacher] + [f"Student_{i}" for i in range(len(students))]

    t_run = teacher_run_dir(teacher_id)
    run_paths = [str(t_run)] + [str(s) for s in students]
    abl_dirs = [str(ablation_idx) if ablation_idx is not None else "null"] + ["null"] * len(students)

    out_dir = OUT_ROOT / f"teacher{teacher_id}_ablation_{direction_key}"
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
    print(f"Teacher: swap_{teacher_id}  Direction: {label_teacher}  Students: {len(students)}")
    print(f"Output:  {out_dir}")
    print(f"{'='*60}")

    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode} — check {log_path}")
    else:
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
    directions = [("no_ablation", None)] + [(str(d), d) for d in range(14)]

    skipped = []
    ran = []

    for teacher_id in TEACHER_SEEDS:
        t_dir = teacher_run_dir(teacher_id)
        if not t_dir.is_dir():
            print(f"\nSkipping teacher swap_{teacher_id} — directory not found.")
            continue

        print(f"\n{'#'*60}")
        print(f"Teacher: swap_{teacher_id}  ({t_dir})")
        print(f"{'#'*60}")

        for direction_key, ablation_idx in directions:
            students = find_students(direction_key, teacher_id)
            key = f"teacher{teacher_id}/{direction_key}"
            if not students:
                skipped.append(key)
                print(f"  Skipping {key} — no students found.")
                continue
            ran.append(key)
            run_sweep(teacher_id, direction_key, ablation_idx, students)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"Ran ({len(ran)}):     {ran}")
    print(f"Skipped ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
