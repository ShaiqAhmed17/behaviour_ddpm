# Quick Start: Neuron Ablation & Model-to-Model Recovery

This is a quick reference for running the complete ablation and recovery workflow.

## Prerequisites

- A trained model (e.g., from your existing training runs)
- Model config file (YAML)
- Model checkpoint file (state.mdl)

## Three Simple Steps

### Step 1: Run Ablation Experiment

```bash
# Edit the file to set your model path
python ablation_example_simple.py
```

**What it does:**
- Loads your trained model
- Ablates neuron #5 (configurable)
- Generates samples from intact and ablated models
- Produces comparison plots and metrics

**Outputs:**
- `ablation_results/intact_samples.pt`
- `ablation_results/ablated_samples.pt`
- `ablation_results/comparison.png`
- `ablation_results/metrics.json`

### Step 2: Analyze the Effect

Look at the comparison plot and metrics to see how much the neuron mattered:

```python
import json
with open('ablation_results/metrics.json') as f:
    metrics = json.load(f)
    
print(f"Mean shift: {metrics['mean_shift']}")
print(f"Std change: {metrics['std_change']}")
```

**Interpretation:**
- `mean_shift < 0.1`: Minimal effect, neuron is redundant
- `mean_shift 0.1-0.5`: Moderate effect, interesting to study
- `mean_shift > 0.5`: Large effect, neuron is functionally important

### Step 3: Train Recovery Model

```bash
python recovery_training.py \
    --ablated_samples ablation_results/ablated_samples.pt \
    --config ddpm/configs/dendritic_hvae/index_cued_first_diffusion_0.3_swap.yaml \
    --output recovery_results \
    --epochs 10000 \
    --validate
```

**What it does:**
- Trains a new model to match the ablated model's behavior
- Saves training progress and checkpoints
- Validates the recovery by comparing distributions

**Outputs:**
- `recovery_results/final_model.pt`
- `recovery_results/training_curve.png`
- `recovery_results/validation_comparison.png`
- `recovery_results/validation_metrics.json`

## Complete Example

Here's a complete workflow with actual paths:

```bash
# 1. Ablate neuron
python ablation_example_simple.py
# → Edit RUN_PATH in the script first!

# 2. Check the effect
cat ablation_results/metrics.json

# 3. If effect is interesting, train recovery model
python recovery_training.py \
    --ablated_samples ablation_results/ablated_samples.pt \
    --config ddpm/configs/dendritic_hvae/index_cued_first_diffusion_0.3_swap.yaml \
    --output recovery_results \
    --epochs 10000 \
    --batch_size 32 \
    --lr 1e-3 \
    --validate

# 4. Compare recovery to target
# Look at recovery_results/validation_comparison.png
```

## Advanced Usage

### Ablate Multiple Neurons

```python
from ablation_example_simple import *

for neuron_idx in [0, 1, 2, 5, 10]:
    print(f"\n{'='*70}")
    print(f"Testing neuron {neuron_idx}")
    print('='*70)
    
    # Load fresh model
    model, task, args = load_model_simple(RUN_PATH, device)
    
    # Ablate
    ablate_neuron_simple(model, neuron_idx)
    
    # Generate samples
    ablated = generate_samples_batch(model, task, 20, device=device)
    
    # Save
    torch.save(ablated, f'ablation_results/ablated_neuron_{neuron_idx}.pt')
```

### Custom Recovery Training

For more control over recovery training:

```python
from recovery_training import build_recovery_model, train_recovery_model

# Build model
model, task, _, mse_pred, mse_target = build_recovery_model(
    'path/to/config.yaml',
    device='cuda'
)

# Load target
target_samples = torch.load('ablation_results/ablated_samples.pt')

# Train with custom parameters
losses = train_recovery_model(
    model=model,
    task=task,
    target_samples=target_samples,
    mse_key_pred=mse_pred,
    mse_key_target=mse_target,
    num_epochs=20000,        # More epochs
    batch_size=64,           # Larger batches
    lr=5e-4,                 # Lower learning rate
    logging_freq=500,
    save_dir='recovery_custom',
    device='cuda'
)
```

## File Organization

After running the complete workflow, you'll have:

```
behaviour_ddpm/
├── ablation_results/
│   ├── intact_samples.pt           # Original model outputs
│   ├── ablated_samples.pt          # Ablated model outputs
│   ├── comparison.png              # Visual comparison
│   └── metrics.json                # Quantitative metrics
│
└── recovery_results/
    ├── final_model.pt              # Trained recovery model
    ├── checkpoint_epoch_*.pt       # Training checkpoints
    ├── training_curve.png          # Loss over time
    ├── training_losses.npy         # Raw loss values
    ├── validation_comparison.png   # Recovery vs target
    └── validation_metrics.json     # Recovery accuracy
```

## Common Issues

### Issue: "Can't find model file"
**Solution:** Update the `RUN_PATH` variable in `ablation_example_simple.py` to point to your trained model directory.

### Issue: "Out of memory"
**Solution:** Reduce batch sizes:
```python
# In ablation_example_simple.py
NUM_BATCHES = 10  # Instead of 20

# In recovery_training.py
--batch_size 16  # Instead of 32
```

### Issue: "No effect from ablation"
**Solution:** 
1. Try different neurons
2. Check if the neuron is actually active (look at activations before ablation)
3. Ablate multiple neurons at once

### Issue: "Recovery model doesn't match target"
**Solution:**
1. Train for more epochs (try 20000-50000)
2. Lower learning rate (try 5e-4 or 1e-4)
3. Check that target distribution is stable (re-run ablation to verify)

## Scientific Workflow

For a proper scientific analysis:

1. **Ablate systematically**: Test all neurons or a representative sample
   
2. **Quantify effects**: Use consistent metrics (mean shift, KL divergence, etc.)
   
3. **Multiple runs**: Run ablation multiple times to check stability
   
4. **Recovery comparison**: Test if different architectures can recover the same behavior
   
5. **Functional interpretation**: Relate ablation effects to task variables

## Example Analysis Script

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load all ablation results
results = {}
for neuron_idx in range(16):  # Assuming 16 neurons
    try:
        samples = torch.load(f'ablation_results/ablated_neuron_{neuron_idx}.pt')
        results[neuron_idx] = samples
    except:
        pass

# Load intact samples
intact = torch.load('ablation_results/intact_samples.pt')
intact_mean = intact.mean(dim=0)

# Compute effect for each neuron
effects = {}
for neuron_idx, ablated in results.items():
    ablated_mean = ablated.mean(dim=0)
    effect = torch.norm(intact_mean - ablated_mean).item()
    effects[neuron_idx] = effect

# Plot
plt.figure(figsize=(10, 6))
plt.bar(effects.keys(), effects.values())
plt.xlabel('Neuron Index')
plt.ylabel('Effect Size (Mean Shift)')
plt.title('Ablation Effect by Neuron')
plt.savefig('neuron_importance.png')
print("Saved importance plot")

# Find most important neurons
sorted_neurons = sorted(effects.items(), key=lambda x: x[1], reverse=True)
print("\nMost important neurons:")
for neuron_idx, effect in sorted_neurons[:5]:
    print(f"  Neuron {neuron_idx}: effect = {effect:.4f}")
```

## Next Steps

After completing this workflow:

1. **Interpret results**: What does the ablated neuron represent functionally?
   
2. **Test hypotheses**: Does the neuron respond to specific task features?
   
3. **Compare architectures**: Can different model architectures achieve the same ablated behavior?
   
4. **Scale up**: Try ablating multiple neurons, neuron populations, or entire layers

For more details, see:
- **Detailed guide**: `ABLATION_GUIDE.md`
- **Full implementation**: `ablation_and_recovery.py`
- **Simple example**: `ablation_example_simple.py`
- **Recovery training**: `recovery_training.py`
