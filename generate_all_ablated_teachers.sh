#!/bin/bash
#
# Generate all 14 ablated teacher models with consistent ablation throughout prep AND diffusion phases.
# These models will be used as teachers for student training.
#

set -e  # Exit on error

SOURCE_MODEL="results_link_sampler/index_cued_first_diffusion_0.3_swap_7"
OUTPUT_BASE="results_link_sampler/ablated_teachers_full"

echo "========================================================================"
echo "GENERATING 14 ABLATED TEACHER MODELS (Prep + Diffusion Ablation)"
echo "========================================================================"
echo "Source model: $SOURCE_MODEL"
echo "Output directory: $OUTPUT_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Generate all 14 ablated models
for neuron in {0..13}; do
    OUTPUT_DIR="${OUTPUT_BASE}/ablation_neuron_${neuron}"
    
    echo "------------------------------------------------------------------------"
    echo "Creating ablated teacher for neuron $neuron"
    echo "------------------------------------------------------------------------"
    
    # Create directory
    mkdir -p "$OUTPUT_DIR"
    
    # Copy original model files
    echo "  Copying model files..."
    cp "${SOURCE_MODEL}/args.yaml" "$OUTPUT_DIR/"
    cp "${SOURCE_MODEL}/state.mdl" "$OUTPUT_DIR/"
    
    # Create ablation marker file with metadata
    echo "  Creating ablation metadata..."
    python3 -c "
import torch
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
sys.path.insert(0, str(REPO_ROOT))

from ddpm.utils.loading import generate_model_and_task_from_args_path_multiepoch

# Load model to get ablation vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args_path = Path('${SOURCE_MODEL}/args.yaml')
checkpoint_path = args_path.parent / 'state.mdl'

args, task, model, _, _ = generate_model_and_task_from_args_path_multiepoch(
    str(args_path), device
)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)

# Get ablation vector (normalized)
ablation_vector = model.behaviour_nullspace[${neuron}].clone()
ablation_vector = ablation_vector / torch.norm(ablation_vector)

# Save ablation data
ablation_data = {
    'neuron_idx': ${neuron},
    'ablation_vector': ablation_vector.cpu(),
    'ablation_type': 'full',  # Applied to both prep AND diffusion
    'source_model': '${SOURCE_MODEL}',
}

output_path = Path('$OUTPUT_DIR') / 'ablation_data.pt'
torch.save(ablation_data, output_path)

print(f'✓ Saved ablation data to {output_path}')
print(f'  Neuron: ${neuron}')
print(f'  Vector shape: {ablation_vector.shape}')
print(f'  Vector norm: {torch.norm(ablation_vector).item():.6f}')
"
    
    echo "  ✓ Created ablated teacher: $OUTPUT_DIR"
    echo ""
done

echo "========================================================================"
echo "✓ ALL 14 ABLATED TEACHERS CREATED"
echo "========================================================================"
echo ""
echo "Teachers saved to: $OUTPUT_BASE"
echo ""
echo "Directory structure:"
ls -d ${OUTPUT_BASE}/ablation_neuron_* | head -5
if [ $(ls -d ${OUTPUT_BASE}/ablation_neuron_* | wc -l) -gt 5 ]; then
    echo "  ... (and $(( $(ls -d ${OUTPUT_BASE}/ablation_neuron_* | wc -l) - 5 )) more)"
fi
echo ""
echo "Each directory contains:"
echo "  - args.yaml (model config)"
echo "  - state.mdl (model weights)"
echo "  - ablation_data.pt (ablation vector + metadata)"
echo ""
echo "Next steps:"
echo "  1. Extract teacher trajectories using extract_ablated_diffusion_states.py"
echo "  2. Train students on these ablated teachers using m-t-m_multiepoch.py"
echo "  3. Compare teacher vs student trajectories"
