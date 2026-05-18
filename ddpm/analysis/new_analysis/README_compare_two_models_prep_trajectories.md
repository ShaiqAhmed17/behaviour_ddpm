# Compare Two Models: Preparatory Neural Trajectories

Notebook: `ddpm/analysis/new_analysis/compare_two_models_prep_trajectories.ipynb`

## Purpose
Compare preparatory neural trajectories between any two models:
- teacher vs student
- teacher vs teacher
- student vs student

Main score is Procrustes distance on the **trial-averaged preparatory trajectory** over 288 non-swap trials.

Secondary scores:
- centroid distance (preparatory point clouds)
- sliced-Wasserstein distance
- RBF MMD

## Input modes
Each model config supports:
1. `source: checkpoint`
- provide `args_path` and `checkpoint_path`
- optional teacher ablation: `nullspace_json_path` + `ablation_direction_idx`

2. `source: array`
- provide `array_path` to `.npy` or `.npz`
- expected shape: `[n_trials, T_prep, D]` or `[T_prep, D]`

## Defaults
- 288 trials generated via angle grid (`ANGLE_STEP=30`)
- all preparatory epochs concatenated (`PREP_EPOCH_INDICES=None`)
- orthogonal Procrustes without scaling (`ALLOW_SCALING_IN_PROCRUSTES=False`)
- optional joint PCA reduction (`PCA_DIMS=10`)

## Suggested run order
1. Run imports/setup cell.
2. Edit `MODEL_A` and `MODEL_B` configs.
3. Run helper + metrics definition cells.
4. Run main comparison cell (prints JSON summary).
5. Run plotting cell.
6. Optionally run synthetic sanity-check cell.

## Notes
- If model A and B have different preparatory trajectory lengths, the notebook raises an error.
- If you need time-warping before Procrustes, add interpolation/DTW before metric computation.
