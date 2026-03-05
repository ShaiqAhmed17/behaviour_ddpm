#!/bin/bash
#
# Extract neural states from all 14 ablated teachers + healthy baseline
# for comparison and visualization
#

set -e

HEALTHY_TEACHER="results_link_sampler/index_cued_first_diffusion_0.3_swap_7"
ABLATED_TEACHERS_BASE="results_link_sampler/ablated_teachers_full"
OUTPUT_DIR="teacher_neural_states"
PREP_IDX=2  # Delay 1 onset

echo "========================================================================"
echo "EXTRACTING NEURAL STATES FROM ALL TEACHERS"
echo "========================================================================"
echo "Prep index: $PREP_IDX (delay 1 onset)"
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Extract healthy baseline
echo "------------------------------------------------------------------------"
echo "Extracting HEALTHY teacher (baseline)"
echo "------------------------------------------------------------------------"
python extract_teacher_states.py \
    --teacher_path "${HEALTHY_TEACHER}/args.yaml" \
    --prep_idx $PREP_IDX \
    --output_path "${OUTPUT_DIR}/healthy_teacher.npz"

echo ""

# Extract all ablated teachers
for neuron in {0..13}; do
    echo "------------------------------------------------------------------------"
    echo "Extracting ablated teacher: neuron $neuron"
    echo "------------------------------------------------------------------------"
    
    python extract_teacher_states.py \
        --teacher_path "${ABLATED_TEACHERS_BASE}/ablation_neuron_${neuron}/args.yaml" \
        --prep_idx $PREP_IDX \
        --output_path "${OUTPUT_DIR}/ablated_teacher_neuron_${neuron}.npz"
    
    echo ""
done

echo "========================================================================"
echo "✓ ALL TEACHER STATES EXTRACTED"
echo "========================================================================"
echo ""
echo "Saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Next: Use the notebook to visualize and compare these states!"
