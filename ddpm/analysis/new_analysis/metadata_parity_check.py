"""Metadata-first parity checks for student/teacher prospective-memory analysis.

This script validates metadata equivalence before any geometry comparison.
It compares trial keys `(cue, color1_angle, color2_angle)` between:
1) deterministic student trial grid, and
2) teacher sweep metadata from saved trajectory artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml


TrialKey = Tuple[int, float, float]


def canonical_angle_deg(angle: float) -> float:
	"""Normalize angle to [0, 360) and round to suppress float noise."""
	return round(float(angle) % 360.0, 6)


def trial_to_key(cue: int, color1_angle: float, color2_angle: float) -> TrialKey:
	return int(cue), canonical_angle_deg(color1_angle), canonical_angle_deg(color2_angle)


def generate_expected_trial_grid(angle_step: int = 30) -> List[TrialKey]:
	if angle_step <= 0 or 360 % angle_step != 0:
		raise ValueError(f"angle_step must divide 360 exactly, got {angle_step}")

	angles = list(range(0, 360, angle_step))
	keys: List[TrialKey] = []
	for cue in (1, 2):
		for color1 in angles:
			for color2 in angles:
				keys.append(trial_to_key(cue, color1, color2))
	return keys


def list_teacher_sweep_files(teacher_run_path: Path) -> List[Path]:
	traj_dir = teacher_run_path / "ablated_teacher_trajectories"
	files = sorted(traj_dir.glob("ablated_teacher_trajectories_sweep_step_*.pt"))
	if not files:
		raise FileNotFoundError(
			f"No sweep files found at {traj_dir}. "
			"Use a run that has initial sweep snapshots."
		)
	return files


def load_teacher_trial_keys(
	files: Sequence[Path],
	expected_set: Optional[set[TrialKey]] = None,
	stop_when_full_coverage: bool = False,
	max_files: Optional[int] = None,
) -> Tuple[List[TrialKey], List[int], int]:
	all_keys: List[TrialKey] = []
	training_steps: List[int] = []
	scanned_files = 0
	seen: set[TrialKey] = set()

	for idx, path in enumerate(files):
		if max_files is not None and idx >= max_files:
			break

		payload = torch.load(path, map_location="cpu", weights_only=False)
		scanned_files += 1
		rows = payload.get("sweep_batch_trials", None)
		if rows is None:
			continue

		step_val = payload.get("training_step", None)
		if step_val is not None:
			training_steps.append(int(step_val))

		for row in rows:
			key = trial_to_key(
				cue=int(row["cue"]),
				color1_angle=float(row["color1_angle"]),
				color2_angle=float(row["color2_angle"]),
			)
			all_keys.append(key)
			seen.add(key)

		if stop_when_full_coverage and expected_set is not None and seen == expected_set:
			break

	if not all_keys:
		raise RuntimeError("Teacher sweep files were found but no sweep_batch_trials were loaded.")

	return all_keys, training_steps, scanned_files


def load_resume_source_from_args(run_path: Path) -> Optional[str]:
	args_path = run_path / "args.yaml"
	if not args_path.exists():
		return None

	with open(args_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	resume_path = cfg.get("resume_path", None)
	if not resume_path:
		return None
	return str(resume_path)


def resolve_run_path(repo_root: Path, run_path_or_name: str) -> Path:
	candidate = Path(run_path_or_name)
	if candidate.is_absolute() and candidate.is_dir():
		return candidate

	if not candidate.is_absolute():
		direct = (repo_root / candidate).resolve()
		if direct.is_dir():
			return direct

		roots = [
			repo_root / "results_link_sampler",
			repo_root / "results_link_sampler_ext",
			repo_root / "results_link_drl",
		]
		for root in roots:
			alt = (root / run_path_or_name).resolve()
			if alt.is_dir():
				return alt

	raise FileNotFoundError(f"Could not resolve run path: {run_path_or_name}")


def summarize_key_counter(counter: Counter, top_k: int = 8) -> List[Dict[str, object]]:
	rows = []
	for (cue, c1, c2), count in counter.most_common(top_k):
		rows.append({
			"cue": cue,
			"color1_angle": c1,
			"color2_angle": c2,
			"count": int(count),
		})
	return rows


@dataclass
class MetadataParityReport:
	teacher_run_path: str
	student_run_path: Optional[str]
	inferred_student_from_teacher_resume: Optional[str]
	inferred_student_matches_provided: Optional[bool]
	angle_step_deg: int
	expected_total_trials: int
	teacher_total_rows: int
	teacher_unique_trials: int
	expected_unique_trials: int
	unique_trial_sets_match_exactly: bool
	teacher_missing_trials_count: int
	teacher_extra_trials_count: int
	teacher_missing_trials_example: List[Dict[str, object]]
	teacher_extra_trials_example: List[Dict[str, object]]
	teacher_cue_distribution: Dict[str, int]
	teacher_bin_coverage_by_cue: Dict[str, int]
	training_step_min: Optional[int]
	training_step_max: Optional[int]
	scanned_sweep_files: int
	total_sweep_files_available: int
	stopped_early_on_full_coverage: bool
	top_repeated_teacher_trials: List[Dict[str, object]]
	strict_pass: bool


def bin_coverage_by_cue(keys: Iterable[TrialKey], angle_step: int) -> Dict[str, int]:
	n_bins = 360 // angle_step
	bins_by_cue = {1: set(), 2: set()}
	for cue, c1, c2 in keys:
		used = c1 if cue == 1 else c2
		b = int(used // angle_step) % n_bins
		bins_by_cue[int(cue)].add(b)
	return {str(cue): len(bins) for cue, bins in bins_by_cue.items()}


def main() -> None:
	parser = argparse.ArgumentParser(description="Metadata-first parity check for student/teacher runs")
	parser.add_argument("--repo_root", type=str, default="/scratch3/shaiq_home/repos/behaviour_ddpm")
	parser.add_argument("--teacher_run", type=str, required=True)
	parser.add_argument("--student_run", type=str, default=None)
	parser.add_argument("--angle_step", type=int, default=30)
	parser.add_argument(
		"--full_scan",
		action="store_true",
		help="Scan all available sweep files instead of stopping after full metadata coverage.",
	)
	parser.add_argument(
		"--max_files",
		type=int,
		default=None,
		help="Optional cap on number of sweep files scanned.",
	)
	parser.add_argument("--out", type=str, default=None)
	args = parser.parse_args()

	repo_root = Path(args.repo_root).resolve()
	teacher_run_path = resolve_run_path(repo_root, args.teacher_run)
	student_run_path = resolve_run_path(repo_root, args.student_run) if args.student_run else None

	expected_keys = generate_expected_trial_grid(angle_step=args.angle_step)
	expected_set = set(expected_keys)

	sweep_files = list_teacher_sweep_files(teacher_run_path)
	teacher_keys, training_steps, scanned_files = load_teacher_trial_keys(
		sweep_files,
		expected_set=expected_set,
		stop_when_full_coverage=(not args.full_scan),
		max_files=args.max_files,
	)
	teacher_counter = Counter(teacher_keys)
	teacher_set = set(teacher_keys)

	missing_trials = sorted(expected_set - teacher_set)
	extra_trials = sorted(teacher_set - expected_set)

	cue_counter = Counter([k[0] for k in teacher_keys])
	inferred_resume = load_resume_source_from_args(teacher_run_path)

	inferred_match: Optional[bool] = None
	if student_run_path is not None and inferred_resume is not None:
		inferred_resume_path = (repo_root / inferred_resume).resolve()
		inferred_match = inferred_resume_path == (student_run_path / "state.mdl").resolve()

	report = MetadataParityReport(
		teacher_run_path=str(teacher_run_path),
		student_run_path=str(student_run_path) if student_run_path else None,
		inferred_student_from_teacher_resume=inferred_resume,
		inferred_student_matches_provided=inferred_match,
		angle_step_deg=int(args.angle_step),
		expected_total_trials=len(expected_keys),
		teacher_total_rows=len(teacher_keys),
		teacher_unique_trials=len(teacher_set),
		expected_unique_trials=len(expected_set),
		unique_trial_sets_match_exactly=(teacher_set == expected_set),
		teacher_missing_trials_count=len(missing_trials),
		teacher_extra_trials_count=len(extra_trials),
		teacher_missing_trials_example=[
			{"cue": c, "color1_angle": c1, "color2_angle": c2}
			for c, c1, c2 in missing_trials[:10]
		],
		teacher_extra_trials_example=[
			{"cue": c, "color1_angle": c1, "color2_angle": c2}
			for c, c1, c2 in extra_trials[:10]
		],
		teacher_cue_distribution={"1": int(cue_counter.get(1, 0)), "2": int(cue_counter.get(2, 0))},
		teacher_bin_coverage_by_cue=bin_coverage_by_cue(teacher_set, angle_step=args.angle_step),
		training_step_min=min(training_steps) if training_steps else None,
		training_step_max=max(training_steps) if training_steps else None,
		scanned_sweep_files=scanned_files,
		total_sweep_files_available=len(sweep_files),
		stopped_early_on_full_coverage=(not args.full_scan and teacher_set == expected_set),
		top_repeated_teacher_trials=summarize_key_counter(teacher_counter, top_k=8),
		strict_pass=(teacher_set == expected_set),
	)

	payload = asdict(report)
	print(json.dumps(payload, indent=2))

	if args.out:
		out_path = Path(args.out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(payload, f, indent=2)


if __name__ == "__main__":
	main()
