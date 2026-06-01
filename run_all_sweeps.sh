#!/bin/bash
# Re-run all compare_model_responses.py sweeps to regenerate plots with updated style.
# Reconstructed from results/all_teacher_student_sweeps/*/run.log
set -euo pipefail
REPO=/scratch3/shaiq_home/repos/behaviour_ddpm
OUT=$REPO/results/all_teacher_student_sweeps
TEACHER=index_cued_first_diffusion_0.3_swap_7
export MPLBACKEND=Agg

run() {
    local dir=$1; shift
    echo "====== $dir ======"
    python compare_model_responses.py "$@" --sweep --student_vs_student \
        --out_dir "$OUT/$dir" 2>&1 | tee "$OUT/$dir/run.log"
}

cd "$REPO"

run ablation_no_ablation \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_no_ablation_0 \
    --labels "Healthy" "Student_0"

run ablation_0 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_0_0 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_0_1 \
    --labels "Ablated_0" "Student_0" "Student_1" \
    --ablation_directions 0 null null

run ablation_1 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_1_0 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_1_1 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_1_2 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_1_3 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_1_4 \
    --labels "Ablated_1" "Student_0" "Student_1" "Student_2" "Student_3" "Student_4" \
    --ablation_directions 1 null null null null null

run ablation_2 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_2_0 \
    --labels "Ablated_2" "Student_0" \
    --ablation_directions 2 null

run ablation_3 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_3_0 \
    --labels "Ablated_3" "Student_0" \
    --ablation_directions 3 null

run ablation_4 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_4_0 \
    --labels "Ablated_4" "Student_0" \
    --ablation_directions 4 null

run ablation_5 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_5_0 \
    --labels "Ablated_5" "Student_0" \
    --ablation_directions 5 null

run ablation_6 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_6_0 \
    --labels "Ablated_6" "Student_0" \
    --ablation_directions 6 null

run ablation_7 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_7_0 \
    --labels "Ablated_7" "Student_0" \
    --ablation_directions 7 null

run ablation_9 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_9_0 \
    --labels "Ablated_9" "Student_0" \
    --ablation_directions 9 null

run ablation_10 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_10_0 \
    --labels "Ablated_10" "Student_0" \
    --ablation_directions 10 null

run ablation_11 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_11_0 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_11_1 \
    --labels "Ablated_11" "Student_0" "Student_1" \
    --ablation_directions 11 null null

run ablation_12 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_12_0 \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_12_1 \
    --labels "Ablated_12" "Student_0" "Student_1" \
    --ablation_directions 12 null null

run ablation_13 \
    --run_paths $TEACHER \
        index_cued_first_diffusion_0.3_swap_recovery_ablation_13_0 \
    --labels "Ablated_13" "Student_0" \
    --ablation_directions 13 null

echo "All sweeps complete."
