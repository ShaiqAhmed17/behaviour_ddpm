#!/bin/bash

# Test extraction script for neuron 0 ablation

MODEL_PATH="results_link_sampler/index_cued_first_diffusion_0.3_swap_7/args.yaml"
OUTPUT_DIR="ablation_diffusion_states"
NEURON_IDX=0

echo "Testing diffusion state extraction..."
echo "Model: $MODEL_PATH"
echo "Ablate neuron: $NEURON_IDX"
echo "Output: $OUTPUT_DIR/neuron_${NEURON_IDX}_states.pt"

python extract_ablated_diffusion_states.py \
    --model_path "$MODEL_PATH" \
    --ablate_neuron $NEURON_IDX \
    --n_trials 24 \
    --output_path "$OUTPUT_DIR/neuron_${NEURON_IDX}_states.pt"

echo "Done! Check the output file."
