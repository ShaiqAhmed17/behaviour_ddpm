"""
Compare ablated vs healthy teacher trajectories to verify ablation is surgical.

Checks if the difference between ablated and healthy teachers lies only in the
ablation direction (behaviour_nullspace[7]).
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "ddpm"))
sys.path.insert(0, str(Path(__file__).parent / "drl"))

from purias_utils.util.arguments_yaml import ConfigNamepace
from ddpm import tasks, model

# ============================================================================
# PART 1: Load paths and checkpoints
# ============================================================================

REPO_ROOT = Path(__file__).parent
ABLATED_TEACHER_PATH = REPO_ROOT / "results_link_drl" / "swap_recovery_6" / "samples" / "step_62_ablated_teacher_trajectories.pt"
TEACHER_CHECKPOINT = REPO_ROOT / "ddpm" / "checkpoints" / "index_cued_first_diffusion_0.3_swap_7.pt"

print("=" * 80)
print("COMPARING ABLATED VS HEALTHY TEACHER TRAJECTORIES")
print("=" * 80)

# Verify files exist
if not ABLATED_TEACHER_PATH.exists():
    print(f"❌ Ablated teacher file not found: {ABLATED_TEACHER_PATH}")
    sys.exit(1)

if not TEACHER_CHECKPOINT.exists():
    print(f"❌ Teacher checkpoint not found: {TEACHER_CHECKPOINT}")
    sys.exit(1)

print(f"✓ Ablated teacher: {ABLATED_TEACHER_PATH.name}")
print(f"✓ Teacher checkpoint: {TEACHER_CHECKPOINT.name}")

# ============================================================================
# PART 2: Load ablated teacher trajectories
# ============================================================================

print("\n[PART 2] Loading ablated teacher trajectories...")
ablated_data = torch.load(str(ABLATED_TEACHER_PATH), map_location="cpu")
ablated_trajectories = ablated_data["ablated_teacher_trajectories"]  # [batch=32, samples=512, T=40, dims=16]
task_info = ablated_data.get("task_variable_information", None)

print(f"  Ablated trajectories shape: {ablated_trajectories.shape}")
print(f"  Task info available: {task_info is not None}")

# ============================================================================
# PART 3: Load behaviour_nullspace from checkpoint
# ============================================================================

print("\n[PART 3] Loading behaviour_nullspace from checkpoint...")
checkpoint = torch.load(str(TEACHER_CHECKPOINT), map_location="cpu")

if "behaviour_nullspace" not in checkpoint:
    print("❌ behaviour_nullspace not found in checkpoint")
    print(f"Available keys: {list(checkpoint.keys())[:10]}")
    sys.exit(1)

behaviour_nullspace = checkpoint["behaviour_nullspace"]  # [14, 16]
print(f"  Behaviour nullspace shape: {behaviour_nullspace.shape}")

# Extract ablation vector (direction 7, unit-normalized)
ablation_vector = behaviour_nullspace[7]  # [16]
ablation_vector = ablation_vector / torch.norm(ablation_vector)

print(f"  Ablation vector (direction 7) norm: {torch.norm(ablation_vector):.4f}")
print(f"  Ablation vector: {ablation_vector.numpy()}")

# Get orthonormal basis
# Ablation direction + 15 orthogonal directions (excluding behavior subspace)
behavior_subspace = behaviour_nullspace[:2]  # Behavior is 2D
behaviour_plane_projection = torch.stack([
    behavior_subspace[0] / torch.norm(behavior_subspace[0]),
    behavior_subspace[1] / torch.norm(behavior_subspace[1])
])

orthogonal_directions = behaviour_nullspace[8:]  # Remaining 6 orthogonal directions (after direction 7)

print(f"  Behavior subspace shape: {behaviour_plane_projection.shape}")
print(f"  Other orthogonal directions: {orthogonal_directions.shape}")

# ============================================================================
# PART 4: Load healthy teacher checkpoint and generate trajectories
# ============================================================================

print("\n[PART 4] Loading healthy teacher model...")

# Load args.yaml to get model config
teacher_dir = TEACHER_CHECKPOINT.parent.parent
args_path = teacher_dir / "args.yaml"

if not args_path.exists():
    print(f"❌ args.yaml not found: {args_path}")
    sys.exit(1)

args = ConfigNamepace.from_yaml_path(str(args_path))
print(f"  Args loaded from: {args_path.name}")

# Create task
task_name = args.task_name
task_config = args.task_config.dict if hasattr(args.task_config, 'dict') else args.task_config
task_obj = getattr(tasks, task_name)(**task_config)

# Setup model parameters
device = 'cpu'
sigma2x_schedule = torch.linspace(
    args.starting_sigma2,
    args.ultimate_sigma2,
    args.num_timesteps,
).to(device)

residual_model_kwargs = args.model_config.dict.pop("residual_model_kwargs").dict
ddpm_model_kwargs = args.model_config.dict.pop("ddpm_model_kwargs").dict
model_builder = getattr(model, args.model_name)

# Handle different task types
try:
    prep_shape = task_obj.sensory_gen.prep_sensory_shape
    underlying_shape = task_obj.sensory_gen.underlying_sensory_shape
except Exception:
    prep_shape = None
    underlying_shape = None

try:
    sample_shape = task_obj.distribution_gen.sample_shape
except Exception:
    sample_shape = task_obj.sample_gen.sample_shape

# Build model
ddpm_model, _, _ = model_builder(
    **args.model_config.dict,
    residual_model_kwargs=residual_model_kwargs,
    ddpm_model_kwargs=ddpm_model_kwargs,
    sigma2x_schedule=sigma2x_schedule,
    prep_sensory_shape=prep_shape,
    underlying_sensory_shape=underlying_shape,
    sample_shape=sample_shape,
    device=device,
)

# Load model state
model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
if model_state is None:
    print("❌ Could not find model state dict in checkpoint")
    sys.exit(1)

ddpm_model.load_state_dict(model_state)
ddpm_model.to(device)
ddpm_model.eval()
print(f"  Model loaded and set to eval mode")

# Get trial information
n_trials = ablated_trajectories.shape[0]
n_samples = ablated_trajectories.shape[1]

print("\n[PART 4b] Generating healthy teacher trajectories...")

# Extract prep and diffusion inputs from task_info
if task_info is None:
    print("⚠️  No task_info available - cannot regenerate on exact same conditions")
    print("    Using random trials instead")
    healthy_trajectories = None
else:
    # Reconstruct the network inputs from task info
    # This depends on the task structure - for cue_first_diffusion_0.3_swap:
    prep_inputs = task_info.get('prep_network_inputs', None)
    diffusion_inputs = task_info.get('diffusion_network_inputs', None)
    prep_epoch_durations = task_info.get('prep_epoch_durations', None)
    diffusion_epoch_durations = task_info.get('diffusion_epoch_durations', None)
    
    if prep_inputs is not None and diffusion_inputs is not None:
        # Move to device
        if isinstance(prep_inputs, list):
            prep_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in prep_inputs]
        elif isinstance(prep_inputs, dict):
            prep_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in prep_inputs.items()}
        else:
            prep_inputs = prep_inputs.to(device) if isinstance(prep_inputs, torch.Tensor) else prep_inputs
            
        if isinstance(diffusion_inputs, list):
            diffusion_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in diffusion_inputs]
        elif isinstance(diffusion_inputs, dict):
            diffusion_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in diffusion_inputs.items()}
        else:
            diffusion_inputs = diffusion_inputs.to(device) if isinstance(diffusion_inputs, torch.Tensor) else diffusion_inputs
        
        # Generate samples
        with torch.no_grad():
            print(f"  Generating {n_trials} trials × {n_samples} samples...")
            prep_dicts, samples_dict = ddpm_model.generate_samples(
                prep_network_inputs=prep_inputs,
                diffusion_network_inputs=diffusion_inputs,
                prep_epoch_durations=prep_epoch_durations,
                diffusion_epoch_durations=diffusion_epoch_durations,
                samples_shape=[n_trials, n_samples],
            )
        
        # Extract trajectories
        if 'sample_trajectory' in samples_dict and samples_dict['sample_trajectory'] is not None:
            healthy_trajectories = samples_dict['sample_trajectory'].detach().cpu()  # [batch, samples, T, 16]
        else:
            print("⚠️  No sample_trajectory in generated samples")
            healthy_trajectories = None
        
        print(f"  Generated trajectories shape: {healthy_trajectories.shape if healthy_trajectories is not None else 'N/A'}")
    else:
        print("⚠️  Could not extract prep/diffusion inputs from task_info")
        healthy_trajectories = None

if healthy_trajectories is None:
    print("❌ Could not generate healthy trajectories")
    sys.exit(1)

# ============================================================================
# PART 5: Compare ablated vs healthy at final timestep
# ============================================================================

print("\n[PART 5] Comparing trajectories (final timestep)...")

# Get final timestep trajectories
ablated_final = ablated_trajectories[:, :, -1, :]  # [32, 512, 16]
healthy_final = healthy_trajectories[:, :, -1, :]  # [32, 512, 16]

# Compute difference
difference = ablated_final - healthy_final  # [32, 512, 16]

print(f"  Ablated final shape: {ablated_final.shape}")
print(f"  Healthy final shape: {healthy_final.shape}")
print(f"  Difference shape: {difference.shape}")

# ============================================================================
# PART 6: Project difference onto ablation direction and orthogonal subspaces
# ============================================================================

print("\n[PART 6] Decomposing differences...")

# Flatten for analysis: [32 * 512, 16]
ablated_flat = ablated_final.reshape(-1, 16)
healthy_flat = healthy_final.reshape(-1, 16)
difference_flat = difference.reshape(-1, 16)

# Project onto ablation direction
ablation_component = torch.matmul(difference_flat, ablation_vector)  # [16384]

# Project onto behavior subspace
behavior_component = torch.matmul(difference_flat, behaviour_plane_projection.T)  # [16384, 2]

# Compute magnitudes
ablation_magnitude = torch.abs(ablation_component)
behavior_magnitude = torch.norm(behavior_component, dim=1)

# Remove orthogonal components (other nullspace directions)
orthogonal_component = difference_flat.clone()
for direction in orthogonal_directions:
    direction = direction / torch.norm(direction)
    proj = torch.matmul(orthogonal_component, direction).unsqueeze(1) * direction.unsqueeze(0)
    orthogonal_component = orthogonal_component - proj

orthogonal_magnitude = torch.norm(orthogonal_component, dim=1)

# Compute total magnitude
total_magnitude = torch.norm(difference_flat, dim=1)

print(f"\n  Analysis across all {len(difference_flat)} samples:")
print(f"    Total difference magnitude:      {total_magnitude.mean():.4f} ± {total_magnitude.std():.4f}")
print(f"    Ablation direction component:   {ablation_magnitude.mean():.4f} ± {ablation_magnitude.std():.4f}")
print(f"      → {(ablation_magnitude.mean() / total_magnitude.mean() * 100):.1f}% of difference")
print(f"    Behavior subspace component:    {behavior_magnitude.mean():.4f} ± {behavior_magnitude.std():.4f}")
print(f"      → {(behavior_magnitude.mean() / total_magnitude.mean() * 100):.1f}% of difference")
print(f"    Other orthogonal component:     {orthogonal_magnitude.mean():.4f} ± {orthogonal_magnitude.std():.4f}")
print(f"      → {(orthogonal_magnitude.mean() / total_magnitude.mean() * 100):.1f}% of difference")

# ============================================================================
# PART 7: Per-trial analysis
# ============================================================================

print("\n[PART 7] Per-trial breakdown (first 8 trials)...")

for trial_idx in range(min(8, 32)):
    trial_diff = difference[trial_idx]  # [512, 16]
    
    # Projections
    trial_ablation = torch.matmul(trial_diff, ablation_vector)
    trial_behavior = torch.matmul(trial_diff, behaviour_plane_projection.T)
    
    trial_ablation_mag = torch.abs(trial_ablation).mean()
    trial_behavior_mag = torch.norm(trial_behavior, dim=1).mean()
    trial_total_mag = torch.norm(trial_diff, dim=1).mean()
    
    ablation_pct = (trial_ablation_mag / trial_total_mag * 100) if trial_total_mag > 0 else 0
    behavior_pct = (trial_behavior_mag / trial_total_mag * 100) if trial_total_mag > 0 else 0
    
    print(f"\n  Trial {trial_idx}:")
    print(f"    Total diff magnitude:  {trial_total_mag:.4f}")
    print(f"    Ablation direction:    {trial_ablation_mag:.4f} ({ablation_pct:.1f}%)")
    print(f"    Behavior subspace:     {trial_behavior_mag:.4f} ({behavior_pct:.1f}%)")

# ============================================================================
# PART 8: Conclusion
# ============================================================================

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

ablation_pct_of_diff = (ablation_magnitude.mean() / total_magnitude.mean() * 100)
behavior_pct_of_diff = (behavior_magnitude.mean() / total_magnitude.mean() * 100)

if ablation_pct_of_diff > 80:
    print(f"✅ ABLATION IS SURGICAL")
    print(f"   The difference between ablated and healthy is {ablation_pct_of_diff:.1f}% along")
    print(f"   the ablation direction, confirming proper ablation.")
    print(f"   Behavior subspace shows {behavior_pct_of_diff:.1f}% difference (near zero).")
else:
    print(f"⚠️  ABLATION MAY NOT BE CLEAN")
    print(f"   Only {ablation_pct_of_diff:.1f}% of difference is along ablation direction.")
    print(f"   {behavior_pct_of_diff:.1f}% is in the behavior subspace (unexpected).")

print("=" * 80)
