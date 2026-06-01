"""compare_model_responses.py

Visualise and compare the sampling behaviour of a set of models (teacher-teacher,
teacher-student, or teacher-multiple-students) in response to specific inputs.

Modes
-----
single  (default)
    Sample `--num_samples` times from every model for a single (cue, color1_deg,
    color2_deg) input, produce a training-style plot, and print pairwise sliced-
    Wasserstein distances.

sweep
    Run the same analysis for all 288 input combinations (2 cues × 12 × 12 colors
    at 30-degree steps).  Only SW scores are stored per trial; plots are generated
    only for a small random subset (`--plot_subset`).

Reference-model mode (teacher + N students)
    Use `--reference_idx 0` (default) to mark one model as the teacher/reference.
    SW is then reported as teacher-vs-each-student.  Pass `--student_vs_student`
    to also compute SW between every pair of students.  Sweep mode additionally
    saves per-trial mean ± std across all teacher-vs-student scores.

Usage examples
--------------
# Compare teacher vs student for one input
python compare_model_responses.py \\
    --run_paths results_link_sampler/teacher_run results_link_sampler/student_run \\
    --labels "Teacher" "Student" \\
    --cue 1 --color1 90 --color2 180 \\
    --num_samples 512 \\
    --out_dir results/comparison

# Ablate teacher direction 0, compare to a student
python compare_model_responses.py \\
    --run_paths results_link_sampler/teacher_run results_link_sampler/student_run \\
    --labels "Teacher (ablated dir 0)" "Student" \\
    --ablation_directions 0 null \\
    --cue 2 --color1 60 --color2 150 \\
    --num_samples 512 \\
    --out_dir results/comparison

# Teacher + 4 students, full sweep, also compute student-vs-student SW
python compare_model_responses.py \\
    --run_paths teacher_run student_run_1 student_run_2 student_run_3 student_run_4 \\
    --labels "Teacher" "Student1" "Student2" "Student3" "Student4" \\
    --reference_idx 0 \\
    --student_vs_student \\
    --sweep --num_samples 512 \\
    --out_dir results/comparison
"""

from __future__ import annotations

import argparse
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from ddpm import model, tasks
from ddpm.utils.vis import symmetrize_and_square_axis, save_figure, save_legend
from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart
from purias_utils.util.arguments_yaml import ConfigNamepace


# ---------------------------------------------------------------------------
# Sliced-Wasserstein
# ---------------------------------------------------------------------------

def _random_unit_directions(dim: int, n_projections: int, rng: np.random.Generator) -> np.ndarray:
    vecs = rng.normal(size=(n_projections, dim)).astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 128,
    seed: int = 42,
) -> float:
    """Sliced-Wasserstein distance between two point clouds [n_samples, D]."""
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
    rng = np.random.default_rng(seed)
    dirs = _random_unit_directions(X.shape[1], n_projections, rng)
    dists = [wasserstein_distance(X @ v, Y @ v) for v in dirs]
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_run_path(run_path: str) -> Path:
    p = Path(run_path)
    if p.is_dir():
        return p
    for root in [Path("results_link_sampler"), Path("results_link_sampler_ext"), Path("results_link_drl")]:
        candidate = root / p
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find run directory: {run_path!r}")


def load_model_from_run(
    run_path: str,
    device: str = "cuda",
    checkpoint_name: str = "state.mdl",
):
    """Load a model and its task from a run directory containing args.yaml."""
    run_path = _resolve_run_path(run_path)
    args = ConfigNamepace.from_yaml_path(str(run_path / "args.yaml"))

    task = getattr(tasks, args.task_name)(**args.task_config.dict)

    sigma2x_schedule = torch.linspace(
        args.starting_sigma2, args.ultimate_sigma2, args.num_timesteps
    ).to(device)

    residual_model_kwargs = args.model_config.dict.pop("residual_model_kwargs").dict
    ddpm_model_kwargs = args.model_config.dict.pop("ddpm_model_kwargs").dict

    try:
        prep_shape = task.sensory_gen.prep_sensory_shape
        underlying_shape = task.sensory_gen.underlying_sensory_shape
    except Exception:
        prep_shape = None
        underlying_shape = None

    try:
        sample_shape = task.distribution_gen.sample_shape
    except Exception:
        sample_shape = task.sample_gen.sample_shape

    ddpm_model, _, _ = getattr(model, args.model_name)(
        **args.model_config.dict,
        residual_model_kwargs=residual_model_kwargs,
        ddpm_model_kwargs=ddpm_model_kwargs,
        sigma2x_schedule=sigma2x_schedule,
        prep_sensory_shape=prep_shape,
        underlying_sensory_shape=underlying_shape,
        sample_shape=sample_shape,
        device=device,
    )

    state_path = run_path / checkpoint_name
    ddpm_model.load_state_dict(
        torch.load(state_path, map_location=device, weights_only=True)
    )
    ddpm_model.to(device)
    ddpm_model.eval()
    return ddpm_model, task, args


def get_ablation_vector(m, direction_idx: int) -> torch.Tensor:
    """Return a normalised nullspace direction from the model's behaviour_nullspace."""
    vec = m.behaviour_nullspace[direction_idx].clone()
    return vec / (torch.norm(vec) + 1e-12)


# ---------------------------------------------------------------------------
# Trial-information helpers
# ---------------------------------------------------------------------------

def build_trial_info(
    task,
    cue: int,
    color1_deg: float,
    color2_deg: float,
    num_samples: int,
):
    """Build a MultiepochTrialInformation for a single, specific (cue, c1, c2) input."""
    probe_features = torch.tensor(
        [[color1_deg, color2_deg]], dtype=torch.float32
    ) * (math.pi / 180.0)
    report_features = probe_features.clone()

    override_stim = {
        "probe_features": probe_features,
        "report_features": report_features,
    }
    override_stim_cart = {
        f"{k}_cart": torch.stack(polar2cart(1.0, v), -1)
        for k, v in override_stim.items()
    }
    override_stim_dict = {
        **override_stim,
        **override_stim_cart,
        "cued_item_idx": torch.tensor([cue - 1], dtype=torch.long),
    }

    task_variable_dict = task.task_variable_gen.generate_variable_dict(
        batch_size=1,
        override_stimulus_features_dict=override_stim_dict,
    )
    trial_info = task.generate_trial_information(
        batch_size=1,
        num_samples=num_samples,
        override_task_variable_information=task_variable_dict,
    )
    return trial_info


def move_trial_inputs_to_device(trial_info, device: str):
    """Return (prep_inputs, diff_inputs) moved to device.

    prep_network_inputs / diffusion_network_inputs are either:
      - a list of tensors (one per epoch) — the multiepoch case
      - a dict of tensors — some other tasks
    """
    def _move_item(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    def _move(inputs):
        if isinstance(inputs, dict):
            return {k: _move_item(v) for k, v in inputs.items()}
        return [_move_item(v) for v in inputs]

    return _move(trial_info.prep_network_inputs), _move(trial_info.diffusion_network_inputs)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def generate_samples_for_model(
    ddpm_model,
    trial_info,
    num_samples: int,
    device: str,
    ablation_vector: Optional[torch.Tensor] = None,
    noise_scaler: float = 1.0,
) -> Tuple[List[Dict], Dict]:
    """Run generate_samples for one model on one trial_info."""
    prep_inputs, diff_inputs = move_trial_inputs_to_device(trial_info, device)

    kwargs = dict(
        prep_network_inputs=prep_inputs,
        diffusion_network_inputs=diff_inputs,
        prep_epoch_durations=trial_info.prep_epoch_durations,
        diffusion_epoch_durations=trial_info.diffusion_epoch_durations,
        samples_shape=[1, num_samples],
        noise_scaler=noise_scaler,
    )
    if ablation_vector is not None:
        kwargs["ablation_vector"] = ablation_vector

    with torch.no_grad():
        prep_dicts, samples_dict = ddpm_model.generate_samples(**kwargs)

    return prep_dicts, samples_dict


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _make_diffusion_cmap(num_timesteps: int):
    magma = plt.get_cmap("magma")
    cNorm = colors.Normalize(vmin=1, vmax=num_timesteps)
    cmap = cmx.ScalarMappable(norm=cNorm, cmap=magma)
    cmap.set_array([])
    return cmap


def _samples_to_2d(samples_dict: Dict, task) -> np.ndarray:
    """Return [N_samples, 2] in display space for overlay plots."""
    s = samples_dict["samples"][0].cpu().numpy()  # [N_samples, sample_dim]
    try:
        lm = task.sample_gen.linking_matrix  # [2, sample_dim]
        return s @ lm.T
    except AttributeError:
        return s[:, :2]


def plot_single_trial(
    trial_info,
    all_labels: List[str],
    all_samples_dicts: List[Dict],
    task,
    num_timesteps: int,
    out_path: str,
    cue: int,
    color1_deg: float,
    color2_deg: float,
    reference_idx: Optional[int] = None,
    sw_scores: Optional[Dict] = None,
):
    """
    Training-style plot for a single trial comparing N models.

    Layout (N = number of models):
        Row 0          : task variables (2 axes) + title/SW annotation
        Rows 1..N      : [generated samples | early x0 preds | traj] per model
        Row N+1        : overlay of all model samples (teacher highlighted)

    reference_idx: index of the reference/teacher model for visual highlighting.
    sw_scores: optional dict (label_A, label_B) -> float to annotate the figure.
    """
    N = len(all_labels)
    n_cols = 4
    n_rows = 1 + N + 1
    figsize = (n_cols * 5, n_rows * 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    cmap = _make_diffusion_cmap(num_timesteps)

    # Colour palette: reference gets black, students get tab10
    tab10 = plt.get_cmap("tab10")
    student_color_idx = 0

    def model_color(i):
        if reference_idx is not None and i == reference_idx:
            return "black"
        nonlocal student_color_idx
        c = tab10(student_color_idx)
        student_color_idx += 1
        return c

    model_colors = [model_color(i) for i in range(N)]

    # --- Row 0: task variables + SW annotation ---
    task.task_variable_gen.display_task_variables(
        trial_info.task_variable_information,
        axes[0, 0],
        axes[0, 1],
        batch_idx=0,
    )
    axes[0, 2].axis("off")
    fig.suptitle(
        f"Cue {cue},  {color1_deg:.0f}$^\\circ$ / {color2_deg:.0f}$^\\circ$",
        y=1.01,
    )

    # SW score table in the 4th column header cell
    if sw_scores:
        sw_text = "Sliced-Wasserstein\n" + "\n".join(
            f"{lA} vs {lB}: {v:.4f}" for (lA, lB), v in sw_scores.items()
        )
        axes[0, 3].text(
            0.05, 0.95, sw_text,
            transform=axes[0, 3].transAxes,
            va="top", ha="left", fontsize=8, family="monospace",
        )
    axes[0, 3].axis("off")

    # --- Rows 1..N: per-model visualisation ---
    for i, (label, samples_dict) in enumerate(zip(all_labels, all_samples_dicts)):
        row = i + 1
        is_ref = (reference_idx is not None and i == reference_idx)

        samples_ax = axes[row, 0]
        early_ax = axes[row, 1]
        traj_ax = axes[row, 2]
        axes[row, 3].axis("off")

        for ax in (samples_ax, early_ax, traj_ax):
            for spine in ax.spines.values():
                spine.set_edgecolor(model_colors[i])
                if is_ref:
                    spine.set_linewidth(2.5)

        label_tex = str(label).replace('_', r'\_')
        samples_ax.set_title("Generated samples")
        samples_ax.set_ylabel(label_tex)
        task.sample_gen.display_samples(
            samples_dict["samples"], samples_ax, batch_idx=0
        )
        samples_ax.set_xlabel(r'$x$ (a.u.)')
        symmetrize_and_square_axis(samples_ax)

        early_ax.set_title(r"Early $x_0$ predictions")
        early_preds = samples_dict.get("early_x0_preds")
        if early_preds is not None and not early_preds.isnan().all():
            task.sample_gen.display_early_x0_pred_timeseries(
                early_preds, early_ax, cmap, batch_idx=0
            )
        else:
            sample_traj = samples_dict.get("sample_trajectory")
            if sample_traj is not None:
                task.sample_gen.display_early_x0_pred_timeseries(
                    sample_traj, early_ax, cmap, batch_idx=0
                )
        early_ax.set_xlabel(r'$x$ (a.u.)')
        early_ax.set_ylabel(r'$y$ (a.u.)')
        symmetrize_and_square_axis(early_ax)

        traj_ax.set_title("Sample trajectory")
        sample_traj = samples_dict.get("sample_trajectory")
        if sample_traj is not None:
            task.sample_gen.display_early_x0_pred_timeseries(
                sample_traj, traj_ax, cmap, batch_idx=0
            )
        traj_ax.set_xlabel(r'$x$ (a.u.)')
        traj_ax.set_ylabel(r'$y$ (a.u.)')
        symmetrize_and_square_axis(traj_ax)

    # --- Final row: overlay (teacher on top) ---
    overlay_ax = axes[N + 1, 0]
    overlay_ax.set_title("Samples overlay")
    overlay_ax.set_xlabel(r'$x$ (a.u.)')
    overlay_ax.set_ylabel(r'$y$ (a.u.)')

    # Draw students first, teacher last so it's on top
    draw_order = list(range(N))
    if reference_idx is not None:
        draw_order = [i for i in draw_order if i != reference_idx] + [reference_idx]

    for i in draw_order:
        label = all_labels[i]
        xy = _samples_to_2d(all_samples_dicts[i], task)
        is_ref = (reference_idx is not None and i == reference_idx)
        overlay_ax.scatter(
            xy[:, 0], xy[:, 1],
            alpha=0.6 if is_ref else 0.25,
            s=4 if is_ref else 2,
            color=model_colors[i],
            label=label,
            zorder=10 if is_ref else 1,
        )

    overlay_ax.add_patch(
        plt.Circle((0, 0), getattr(task.sample_gen, "sample_radius", 1.0), color="red", fill=False)
    )
    overlay_ax.legend(markerscale=5)
    symmetrize_and_square_axis(overlay_ax)

    # If reference + multiple students: add a second panel with student density only
    if reference_idx is not None and N > 2:
        student_ax = axes[N + 1, 1]
        student_ax.set_title("Students only (density)")
        ref_xy = _samples_to_2d(all_samples_dicts[reference_idx], task)
        student_ax.scatter(ref_xy[:, 0], ref_xy[:, 1], alpha=0.6, s=4, color="black", label=all_labels[reference_idx], zorder=10)
        for i in range(N):
            if i == reference_idx:
                continue
            xy = _samples_to_2d(all_samples_dicts[i], task)
            student_ax.scatter(xy[:, 0], xy[:, 1], alpha=0.15, s=2, color=model_colors[i])
        student_ax.add_patch(
            plt.Circle((0, 0), getattr(task.sample_gen, "sample_radius", 1.0), color="red", fill=False)
        )
        symmetrize_and_square_axis(student_ax)
        axes[N + 1, 2].axis("off")
    else:
        axes[N + 1, 1].axis("off")
        axes[N + 1, 2].axis("off")

    axes[N + 1, 3].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_figure(fig, Path(out_path))
    save_legend(overlay_ax, Path(out_path))
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# SW score computation
# ---------------------------------------------------------------------------

def _extract_samples_np(all_samples_dicts: List[Dict]) -> List[np.ndarray]:
    """Return a list of [N_samples, sample_dim] float64 arrays, one per model."""
    return [sd["samples"][0].cpu().float().numpy() for sd in all_samples_dicts]


def compute_sw_scores(
    all_labels: List[str],
    all_samples_dicts: List[Dict],
    reference_idx: Optional[int] = None,
    student_vs_student: bool = False,
    n_projections: int = 128,
    seed: int = 42,
) -> Dict[Tuple[str, str], float]:
    """Compute sliced-Wasserstein distances between model distributions.

    When `reference_idx` is set:
      - Always computes reference-vs-each-other-model.
      - Also computes all student pairs when `student_vs_student=True`.
    When `reference_idx` is None:
      - Computes all pairs (original behaviour).

    Returns a dict mapping (label_A, label_B) -> SW distance.
    """
    all_samples = _extract_samples_np(all_samples_dicts)

    if reference_idx is None:
        # All pairs
        pairs = list(combinations(range(len(all_labels)), 2))
    else:
        ref = reference_idx
        others = [i for i in range(len(all_labels)) if i != ref]
        # Reference vs every other model
        pairs = [(ref, i) for i in others]
        if student_vs_student:
            pairs += list(combinations(others, 2))

    scores = {}
    for i, j in pairs:
        sw = sliced_wasserstein_distance(
            all_samples[i], all_samples[j],
            n_projections=n_projections,
            seed=seed,
        )
        scores[(all_labels[i], all_labels[j])] = sw
    return scores


# ---------------------------------------------------------------------------
# Sweep trial generation
# ---------------------------------------------------------------------------

def generate_sweep_trials(angle_step: int = 30) -> List[Dict]:
    """288 = 2 cues × 12 colors × 12 colors at angle_step=30."""
    if 360 % angle_step != 0:
        raise ValueError(f"angle_step must divide 360, got {angle_step}")
    angles = list(range(0, 360, angle_step))
    trials = []
    for cue in [1, 2]:
        for c1 in angles:
            for c2 in angles:
                trials.append({"cue": cue, "color1_deg": float(c1), "color2_deg": float(c2)})
    return trials


# ---------------------------------------------------------------------------
# Main modes
# ---------------------------------------------------------------------------

def _ref_idx(args) -> Optional[int]:
    """Return the reference index, or None if the user disabled it with -1."""
    return None if args.reference_idx == -1 else args.reference_idx


def run_single_trial(
    args,
    all_models,
    all_labels: List[str],
    all_ablation_vectors,
    task,
    device: str,
    out_dir: Path,
):
    num_timesteps = all_models[0].T
    ref_idx = _ref_idx(args)

    trial_info = build_trial_info(
        task,
        cue=args.cue,
        color1_deg=args.color1,
        color2_deg=args.color2,
        num_samples=args.num_samples,
    )

    all_samples_dicts = []
    for m, abl_vec in zip(all_models, all_ablation_vectors):
        _, sd = generate_samples_for_model(m, trial_info, args.num_samples, device, abl_vec)
        all_samples_dicts.append(sd)

    sw_scores = compute_sw_scores(
        all_labels, all_samples_dicts,
        reference_idx=ref_idx,
        student_vs_student=args.student_vs_student,
        n_projections=args.n_sw_projections,
        seed=args.seed,
    )

    plot_name = f"cue{args.cue}_c1{args.color1:.0f}_c2{args.color2:.0f}.png"
    plot_single_trial(
        trial_info=trial_info,
        all_labels=all_labels,
        all_samples_dicts=all_samples_dicts,
        task=task,
        num_timesteps=num_timesteps,
        out_path=str(out_dir / plot_name),
        cue=args.cue,
        color1_deg=args.color1,
        color2_deg=args.color2,
        reference_idx=ref_idx,
        sw_scores=sw_scores,
    )

    print("\nSliced-Wasserstein distances:")
    for (lA, lB), sw in sw_scores.items():
        print(f"  {lA!r} vs {lB!r}: {sw:.6f}")

    if ref_idx is not None:
        ref_label = all_labels[ref_idx]
        ref_vs_students = {lB: v for (lA, lB), v in sw_scores.items() if lA == ref_label}
        ref_vs_students.update({lA: v for (lA, lB), v in sw_scores.items() if lB == ref_label})
        if len(ref_vs_students) > 1:
            vals = list(ref_vs_students.values())
            print(f"\nTeacher ({ref_label}) vs students — mean: {np.mean(vals):.6f}  std: {np.std(vals):.6f}")

    row = {
        "cue": args.cue,
        "color1_deg": args.color1,
        "color2_deg": args.color2,
        "same_color": abs(args.color1 - args.color2) % 360 < 1e-3,
    }
    for (lA, lB), sw in sw_scores.items():
        row[f"sw_{lA}_vs_{lB}"] = sw
    pd.DataFrame([row]).to_csv(out_dir / "sw_single_trial.csv", index=False)
    print(f"Saved SW scores: {out_dir / 'sw_single_trial.csv'}")


def run_sweep(
    args,
    all_models,
    all_labels: List[str],
    all_ablation_vectors,
    task,
    device: str,
    out_dir: Path,
):
    num_timesteps = all_models[0].T
    ref_idx = _ref_idx(args)
    trials = generate_sweep_trials(args.angle_step)
    n_same_color = sum(1 for t in trials if t["color1_deg"] == t["color2_deg"])
    print(
        f"Sweep: {len(trials)} trials "
        f"({n_same_color} with same color, per {360 // args.angle_step}^2 × 2 cues)"
    )
    if ref_idx is not None:
        print(f"Reference model: [{ref_idx}] {all_labels[ref_idx]}")
        student_labels = [l for i, l in enumerate(all_labels) if i != ref_idx]
        print(f"Student models: {student_labels}")

    rng = np.random.default_rng(args.seed)
    plot_indices = set(
        rng.choice(len(trials), size=min(args.plot_subset, len(trials)), replace=False).tolist()
    ) if args.plot_subset > 0 else set()

    rows = []
    for trial_idx, trial in enumerate(trials):
        cue, c1, c2 = trial["cue"], trial["color1_deg"], trial["color2_deg"]
        print(f"  [{trial_idx + 1}/{len(trials)}] cue={cue} c1={c1:.0f}° c2={c2:.0f}°", end="")

        trial_info = build_trial_info(task, cue=cue, color1_deg=c1, color2_deg=c2, num_samples=args.num_samples)

        all_samples_dicts = []
        for m, abl_vec in zip(all_models, all_ablation_vectors):
            _, sd = generate_samples_for_model(m, trial_info, args.num_samples, device, abl_vec)
            all_samples_dicts.append(sd)

        sw_scores = compute_sw_scores(
            all_labels, all_samples_dicts,
            reference_idx=ref_idx,
            student_vs_student=args.student_vs_student,
            n_projections=args.n_sw_projections,
            seed=args.seed + trial_idx,
        )
        sw_summary = "  ".join(f"{lB}:{sw:.3f}" for (lA, lB), sw in sw_scores.items())

        # Self-SW baseline: sample the reference model a second time and compute SW(batch1, batch2)
        sw_self = None
        if ref_idx is not None:
            ref_model = all_models[ref_idx]
            ref_abl_vec = all_ablation_vectors[ref_idx]
            _, ref_sd2 = generate_samples_for_model(ref_model, trial_info, args.num_samples, device, ref_abl_vec)
            ref_samples_1 = all_samples_dicts[ref_idx]["samples"][0].cpu().float().numpy()
            ref_samples_2 = ref_sd2["samples"][0].cpu().float().numpy()
            sw_self = sliced_wasserstein_distance(
                ref_samples_1, ref_samples_2,
                n_projections=args.n_sw_projections,
                seed=args.seed + trial_idx + 10000,
            )
            print(f"  {sw_summary}  [self:{sw_self:.3f}]")
        else:
            print(f"  {sw_summary}")

        row = {
            "trial_idx": trial_idx,
            "cue": cue,
            "color1_deg": c1,
            "color2_deg": c2,
            "same_color": c1 == c2,
        }
        for (lA, lB), sw in sw_scores.items():
            row[f"sw_{lA}_vs_{lB}"] = sw
        if sw_self is not None:
            row["sw_self_reference"] = sw_self

        # Aggregate teacher-vs-students stats (only when reference is set and there are >1 students)
        if ref_idx is not None:
            ref_label = all_labels[ref_idx]
            ref_vs = [v for (lA, lB), v in sw_scores.items()
                      if lA == ref_label or lB == ref_label]
            if len(ref_vs) > 1:
                row["teacher_vs_students_mean"] = float(np.mean(ref_vs))
                row["teacher_vs_students_std"] = float(np.std(ref_vs))

        rows.append(row)

        if trial_idx in plot_indices:
            plot_dir = out_dir / "sweep_plots"
            plot_dir.mkdir(exist_ok=True)
            plot_name = f"trial{trial_idx:04d}_cue{cue}_c1{c1:.0f}_c2{c2:.0f}.png"
            plot_single_trial(
                trial_info=trial_info,
                all_labels=all_labels,
                all_samples_dicts=all_samples_dicts,
                task=task,
                num_timesteps=num_timesteps,
                out_path=str(plot_dir / plot_name),
                cue=cue,
                color1_deg=c1,
                color2_deg=c2,
                reference_idx=ref_idx,
                sw_scores=sw_scores,
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sw_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep SW scores: {csv_path}")

    # Summary stats
    sw_cols = [c for c in df.columns if c.startswith("sw_")]
    print("\nMean sliced-Wasserstein per pair (overall / same-color / different-color):")
    for col in sw_cols:
        same = df.loc[df["same_color"], col].mean()
        diff = df.loc[~df["same_color"], col].mean()
        print(f"  {col}: {df[col].mean():.6f}  same={same:.6f}  diff={diff:.6f}")

    if "sw_self_reference" in df.columns:
        self_mean = df["sw_self_reference"].mean()
        self_same = df.loc[df["same_color"], "sw_self_reference"].mean()
        self_diff = df.loc[~df["same_color"], "sw_self_reference"].mean()
        print(f"\nReference self-SW (noise floor) — overall: {self_mean:.6f}  same: {self_same:.6f}  diff: {self_diff:.6f}")
        print("\nNormalised SW (ratio to self-SW baseline):")
        for col in sw_cols:
            ratio_col = df[col] / df["sw_self_reference"]
            same_ratio = (df.loc[df["same_color"], col] / df.loc[df["same_color"], "sw_self_reference"]).mean()
            diff_ratio = (df.loc[~df["same_color"], col] / df.loc[~df["same_color"], "sw_self_reference"]).mean()
            print(f"  {col}: {ratio_col.mean():.3f}×  same={same_ratio:.3f}×  diff={diff_ratio:.3f}×")

    if "teacher_vs_students_mean" in df.columns:
        print("\nTeacher-vs-students aggregate (mean over student SW per trial):")
        print(f"  overall mean: {df['teacher_vs_students_mean'].mean():.6f}")
        print(f"  same-color:   {df.loc[df['same_color'], 'teacher_vs_students_mean'].mean():.6f}")
        print(f"  diff-color:   {df.loc[~df['same_color'], 'teacher_vs_students_mean'].mean():.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_nullable_int(s: str) -> Optional[int]:
    if s.lower() in ("none", "null", ""):
        return None
    return int(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Models
    p.add_argument("--run_paths", nargs="+", required=True,
                   help="Run directories (or names relative to results_link_sampler) for each model")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display labels for each model (default: directory basename)")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="Checkpoint filename inside each run directory (default: state.mdl for all)")
    p.add_argument("--ablation_directions", nargs="+", default=None,
                   help="Nullspace direction index to ablate per model, or 'null' to disable (default: no ablation)")
    p.add_argument("--reference_idx", type=int, default=0,
                   help="Index of the reference/teacher model for highlighting and directed SW (default: 0). "
                        "Set to -1 to treat all models equally (all-pairs SW).")
    p.add_argument("--student_vs_student", action="store_true",
                   help="Also compute SW between every pair of non-reference (student) models")

    # Mode
    p.add_argument("--sweep", action="store_true",
                   help="Run all 288 input combinations instead of a single trial")
    p.add_argument("--angle_step", type=int, default=30,
                   help="Colour angle step in degrees for sweep mode (default: 30)")
    p.add_argument("--plot_subset", type=int, default=6,
                   help="Number of sweep trials to also plot (default: 6, 0 = none)")

    # Single-trial inputs (ignored in sweep mode)
    p.add_argument("--cue", type=int, default=1, help="Cue index (1 or 2)")
    p.add_argument("--color1", type=float, default=0.0, help="Color 1 angle in degrees")
    p.add_argument("--color2", type=float, default=90.0, help="Color 2 angle in degrees")

    # Sampling
    p.add_argument("--num_samples", type=int, default=512, help="Samples per model per trial")
    p.add_argument("--noise_scaler", type=float, default=1.0)

    # Metrics
    p.add_argument("--n_sw_projections", type=int, default=128,
                   help="Number of random projections for sliced-Wasserstein")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--out_dir", type=str, default="results/model_comparison",
                   help="Directory to save plots and CSVs")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: cuda if available, else cpu)")

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(args.run_paths)

    # Resolve labels
    labels = args.labels if args.labels else [Path(p).name for p in args.run_paths]
    if len(labels) != n_models:
        raise ValueError(f"--labels count ({len(labels)}) must match --run_paths count ({n_models})")

    # Resolve checkpoints
    checkpoints = args.checkpoints if args.checkpoints else ["state.mdl"] * n_models
    if len(checkpoints) != n_models:
        raise ValueError(f"--checkpoints count must match --run_paths count")

    # Resolve ablation directions
    if args.ablation_directions is None:
        ablation_dirs = [None] * n_models
    else:
        if len(args.ablation_directions) != n_models:
            raise ValueError(f"--ablation_directions count must match --run_paths count")
        ablation_dirs = [_parse_nullable_int(s) for s in args.ablation_directions]

    # Load models and task (task loaded from first model; all should share the same task)
    print("Loading models...")
    all_models = []
    task = None
    for i, (run_path, ckpt) in enumerate(zip(args.run_paths, checkpoints)):
        m, t, _ = load_model_from_run(run_path, device=device, checkpoint_name=ckpt)
        all_models.append(m)
        if task is None:
            task = t
        print(f"  [{i}] {labels[i]}  ({run_path})")

    # Build ablation vectors
    all_ablation_vectors = []
    for i, (m, abl_dir) in enumerate(zip(all_models, ablation_dirs)):
        if abl_dir is not None:
            vec = get_ablation_vector(m, abl_dir).to(device)
            print(f"  [{i}] {labels[i]}: ablating nullspace direction {abl_dir}")
            all_ablation_vectors.append(vec)
        else:
            all_ablation_vectors.append(None)

    if args.sweep:
        run_sweep(args, all_models, labels, all_ablation_vectors, task, device, out_dir)
    else:
        run_single_trial(args, all_models, labels, all_ablation_vectors, task, device, out_dir)


if __name__ == "__main__":
    main()
