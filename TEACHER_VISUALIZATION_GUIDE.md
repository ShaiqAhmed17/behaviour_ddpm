# Teacher Visualization & Student Training Guide

## What We've Done

### 1. ✓ Generated 14 Ablated Teachers
- **Location**: `results_link_sampler/ablated_teachers_full/`
- **What**: 14 teacher models with ablation applied throughout BOTH prep and diffusion phases
- **Structure**: Each directory has:
  - `args.yaml` (model config)
  - `state.mdl` (weights - same as healthy)
  - `ablation_data.pt` (ablation vector + metadata)

### 2. ✓ Modified Model Code
- **Files changed**: 
  - `ddpm/model/main/base/linear_subspace.py` (prep-phase ablation)
  - `ddpm/model/main/multiepoch.py` (pass ablation through)
  - `ddpm/model/main/base/hvae.py` (diffusion-phase ablation - already done)
- **Result**: Ablation now applied consistently throughout the entire forward pass

### 3. ✓ Created Extraction Tools
- `extract_teacher_states.py` - Extract neural states from any teacher
- `extract_ablated_diffusion_states.py` - Extract full diffusion trajectories
- `test_prep_ablation.py` - Verify ablation is working

## Visualizing Teachers (SAME as Your Notebook)

### Option A: Use Existing Notebook Directly

Your `prospective_memory_analysis.ipynb` **already works** with ablated teachers! Just change the model path:

```python
# In Cell 2 of notebook, change:
ARGS_PATH = REPO_ROOT / "results_link_sampler/ablated_teachers_full/ablation_neuron_0/args.yaml"
CHECKPOINT_PATH = ARGS_PATH.parent / "state.mdl"
```

The notebook will:
1. Auto-detect `ablation_data.pt`
2. Apply ablation during sampling
3. Generate all the same visualizations (PCA, 2D binning, etc.)

### Option B: Extract All Teachers and Compare

1. **Extract neural states from all teachers**:
   ```bash
   chmod +x extract_all_teacher_states.sh
   conda activate ddpm
   ./extract_all_teacher_states.sh
   ```
   
   This creates: `teacher_neural_states/healthy_teacher.npz` and 14 `ablated_teacher_neuron_X.npz` files

2. **Add comparison cells to notebook**:
   Copy code from `notebook_cells_teacher_comparison.py` into new cells in your notebook. This will:
   - Load healthy + all 14 ablated teachers
   - Compare their neural states
   - Rank neurons by ablation impact
   - Show side-by-side PCA plots

## Training Students on Ablated Teachers

### Single Student Training

```bash
conda activate ddpm

python m-t-m_multiepoch.py \
    configs/your_student_config.yaml \
    --source_run_path results_link_sampler/ablated_teachers_full/ablation_neuron_0 \
    --noise_scaler 1.0
```

**Key Point**: The teacher will automatically apply ablation during sampling (reads from `ablation_data.pt`)

### Train All 14 Students (Batch)

Create a script like:

```bash
#!/bin/bash
for neuron in {0..13}; do
    echo "Training student on ablated teacher: neuron $neuron"
    
    python m-t-m_multiepoch.py \
        configs/your_config.yaml \
        --source_run_path results_link_sampler/ablated_teachers_full/ablation_neuron_${neuron} \
        --save_dir student_models/student_neuron_${neuron} \
        --noise_scaler 1.0
    
    echo "✓ Student $neuron complete"
done
```

## Comparing Teacher vs Student

### After training, extract student states:

```bash
python extract_teacher_states.py \
    --teacher_path student_models/student_neuron_0/args.yaml \
    --prep_idx 2 \
    --output_path student_neural_states/student_neuron_0.npz
```

### Then compare in notebook:

```python
# Load teacher (ablated)
teacher_data = np.load('teacher_neural_states/ablated_teacher_neuron_0.npz')
teacher_states = teacher_data['neural_states']

# Load student (trained on ablated teacher)
student_data = np.load('student_neural_states/student_neuron_0.npz')
student_states = student_data['neural_states']

# Compare
diff = np.linalg.norm(teacher_states - student_states, axis=1).mean()
print(f"Teacher-Student L2 distance: {diff:.4f}")

# Do PCA on both and plot side-by-side
```

## Key Files Summary

| File | Purpose |
|------|---------|
| `generate_all_ablated_teachers.sh` | Create 14 ablated teachers |
| `extract_teacher_states.py` | Extract prep states from any model |
| `extract_ablated_diffusion_states.py` | Extract full diffusion trajectories |
| `extract_all_teacher_states.sh` | Batch extract all 14 teachers |
| `test_prep_ablation.py` | Verify ablation works in prep phase |
| `notebook_cells_teacher_comparison.py` | Example cells for comparing teachers |

## What Changed in the Code

| File | What Changed |
|------|--------------|
| `linear_subspace.py` | Added `ablation_vector` param to `prepare()`, applies ablation after each prep step |
| `multiepoch.py` | Passes `ablation_vector` to all `prepare()` calls |
| `hvae.py` | Already had diffusion ablation (you did this) |

## Verification

To verify everything is working:

```bash
# Test prep-phase ablation
python test_prep_ablation.py

# Should show:
# ✓ SUCCESS: Prep-phase ablation is working!
#   - Ablated component ~0: 0.000008 < 0.1
#   - Prep states differ: 393.05 > 1.0
```

## Next Steps

1. ✓ Teachers generated → **Done!**
2. **Extract and visualize teacher states** → Run `extract_all_teacher_states.sh`
3. **Train students** → Use `m-t-m_multiepoch.py` with ablated teachers
4. **Compare teacher vs student** → Extract student states, compare in notebook

The key insight: With ablation applied throughout (prep + diffusion), the student will learn from a teacher whose memory representation is genuinely altered from the start, not just during output generation!
