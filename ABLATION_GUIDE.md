# Neuron Ablation and Model-to-Model Recovery

This guide explains how to perform neuron ablation experiments and model-to-model recovery in your trained DDPM models.

## Overview

The workflow consists of three main phases:

1. **Neuron Ablation**: Remove a neuron from a trained model
2. **Output Collection**: Generate samples from the ablated model to characterize the new output distribution
3. **Model-to-Model Recovery**: Train a new model to match the ablated model's behavior

## Conceptual Logic

### Why Ablate Neurons?

Neuron ablation is a classic technique in neuroscience and neural network analysis to understand:
- **Functional role**: What does this neuron contribute to the overall computation?
- **Redundancy**: Can the network compensate for the loss?
- **Specialization**: Is the neuron critical for specific task conditions?

### What is Model-to-Model Recovery?

Model-to-model recovery asks: *"Can we train a new model (potentially with different architecture) to reproduce the ablated model's behavior?"*

This is useful for:
- Understanding what minimal architecture is needed for reduced functionality
- Testing if the ablated behavior represents a valid computational strategy
- Discovering whether the original network's structure was necessary

## Technical Details

### 1. Neuron Ablation Methods

There are two primary ways to ablate a neuron:

#### Method A: Weight-Based Ablation (Recommended)
```python
# Zero out incoming weights (prevents neuron from receiving input)
layer.weight[:, neuron_idx] = 0.0

# Zero out outgoing weights (prevents neuron from affecting downstream)
layer.weight[neuron_idx, :] = 0.0
layer.bias[neuron_idx] = 0.0  # if bias exists
```

**Pros**: 
- Permanent modification
- Neuron is truly "removed" from computation
- Easy to save and compare ablated models

**Cons**:
- Requires understanding model architecture
- May need to handle skip connections or residuals

#### Method B: Activity-Based Ablation
```python
def ablation_hook(module, input, output):
    output[..., neuron_idx] = 0.0  # Clamp activity to zero
    return output

handle = model.register_forward_hook(ablation_hook)
```

**Pros**:
- No weight modification needed
- Can be dynamically enabled/disabled
- Works with any architecture

**Cons**:
- Requires keeping the hook active
- Slightly more overhead during inference

### 2. Output Distribution Collection

After ablation, you need to characterize how the model's output distribution changed:

```python
# Generate many samples
num_trials = 1000
samples_per_trial = 512
total_samples = num_trials * samples_per_trial  # e.g., 512,000 samples

# For each trial:
#   1. Generate task input
#   2. Run model inference
#   3. Collect output samples
```

**Key metrics to track**:
- **Mean shift**: How much did the average output move?
- **Variance change**: Did outputs become more/less variable?
- **Distributional shape**: Did the distribution's shape change (e.g., bimodal → unimodal)?

For 2D outputs (common in behavior tasks):
```python
intact_mean = intact_samples.mean(dim=0)  # [2]
ablated_mean = ablated_samples.mean(dim=0)  # [2]
mean_shift = torch.norm(intact_mean - ablated_mean)  # scalar
```

### 3. Model-to-Model Recovery

The recovery phase trains a new model to match the ablated output distribution. This follows the same training procedure as the original model, but with the ablated distribution as the target.

#### Training Loop

```python
for epoch in range(num_epochs):
    # CRITICAL: Sample from ablated distribution (NOT from task!)
    # This is the key difference from normal training
    target_batch = ablated_samples[random_indices]
    
    # Generate task inputs (cues, stimuli, etc.)
    # We still use the task for inputs, just not for target behavior
    task_vars = task.generate()
    sensory_input = task.get_sensory_input(**task_vars)
    
    # Forward diffusion: noise the target samples
    noise_dict = model.noise(target_batch, sensory_input)
    x_t = noise_dict['x_t']  # Noised samples at various timesteps
    
    # Reverse diffusion: predict the noise/score
    pred_dict = model.residual(x_t, sensory_input)
    
    # Compute loss
    loss = MSE(pred_dict['epsilon_hat'], noise_dict['epsilon'])
    
    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Key Insight

The recovery model learns to denoise samples that, when fully denoised, match the ablated distribution. **The crucial difference from normal training**: 
- **Normal training**: Task generates both inputs AND target behavior
- **Recovery training**: Task generates inputs, ablated samples provide target behavior

This means the task config can stay the same - we're not changing the task structure, just what behavior we're trying to reproduce.

## File Descriptions

### `ablation_example_simple.py`
A streamlined, runnable example that:
- Loads your trained model
- Ablates a single neuron
- Generates samples from intact and ablated models
- Compares and visualizes the distributions

**Usage**:
```bash
# Edit the RUN_PATH variable to point to your trained model
python ablation_example_simple.py
```

### `ablation_and_recovery.py`
Complete implementation with classes for:
- `NeuronAblation`: Handles ablation operations
- `AblationDataCollector`: Generates and stores samples
- `ModelToModelRecovery`: Trains recovery models
- Helper functions for loading models and comparing distributions

**Usage**:
```python
from ablation_and_recovery import main_ablation_workflow

main_ablation_workflow(
    config_path="path/to/config.yaml",
    checkpoint_path="path/to/state.mdl",
    neuron_idx_to_ablate=5,
    num_trials=1000,
    num_samples_per_trial=512,
    output_dir='results_ablation_neuron_5'
)
```

## Workflow Example

### Step-by-Step

1. **Train your original model** (you've already done this)
   ```bash
   python ddpm/train/multiepoch.py --config configs/my_config.yaml
   ```

2. **Run ablation experiment**
   ```bash
   python ablation_example_simple.py
   ```
   
   This produces:
   - `ablation_results/intact_samples.pt`: Samples from intact model
   - `ablation_results/ablated_samples.pt`: Samples from ablated model
   - `ablation_results/comparison.png`: Visualization
   - `ablation_results/metrics.json`: Quantitative comparison

3. **Analyze the effect**
   
   Look at the comparison plot and metrics:
   - Small mean_shift (<0.1): Neuron has minimal effect
   - Large mean_shift (>0.5): Neuron is functionally important
   - Changes in distribution shape: Neuron affects task strategy

4. **Perform model-to-model recovery** 
   
   ```python
   from ablation_and_recovery import ModelToModelRecovery
   
   # Load ablated samples
   ablated_samples = torch.load('ablation_results/ablated_samples.pt')
   
   # Train recovery model
   recovery = ModelToModelRecovery(
       target_samples=ablated_samples,
       task=task,
       model_config=config,
       device='cuda'
   )
   
   recovery.train_recovery(
       num_epochs=10000,
       save_dir='recovery_results'
   )
   ```

5. **Compare recovery model to ablated model**
   
   Generate samples from the recovery model and verify it matches the ablated distribution.

## Architecture-Specific Notes

### For VectoralResidualModel

`VectoralResidualModel` has a simple feedforward structure:

```
Input → Linear → Softplus → Linear → ... → Output
```

Ablation targets the state space neurons:
- State space size: e.g., 16 neurons
- Hidden layers: e.g., [128, 128, 128]

**Where to ablate**: In the state space dimension, not the hidden layers.

```python
# State space is preserved through the network
# Ablate in state_space_size dimension
neuron_idx = 5  # Must be < state_space_size
```

### For Dendritic/Axonal Models

For hierarchical models (DendriticResidualModel, AxonalResidualModel), you need to consider:
- Which layer to ablate (proximal vs distal)
- Whether to ablate entire branches
- How ablation propagates through the tree

### For Multi-Epoch Models

Your models often have multiple epochs (preparation, sample):
- Consider ablating during specific epochs
- Track which epoch is most affected by ablation
- Recovery can target specific epoch behavior

## Expected Results

### Typical Ablation Effects

1. **Minimal Impact** (mean_shift < 0.1)
   - Neuron is redundant or inactive
   - Network has learned robust representations
   - Other neurons compensate

2. **Moderate Impact** (0.1 < mean_shift < 0.5)
   - Neuron contributes but isn't critical
   - Some behavioral degradation
   - Interesting target for recovery experiments

3. **Large Impact** (mean_shift > 0.5)
   - Neuron is functionally important
   - Significant behavioral change
   - May reveal specialization for task features

### Recovery Success Metrics

After training a recovery model:
- **Perfect recovery**: Recovery model matches ablated distribution (KL divergence < 0.01)
- **Partial recovery**: Recovery model approximates ablated behavior but with differences
- **Failed recovery**: Recovery model converges to original intact behavior (suggesting the ablated behavior is unstable/unlearnable)

## Scientific Questions to Explore

1. **Redundancy**: How many neurons can be ablated before behavior significantly changes?

2. **Specialization**: Are certain neurons specialized for specific task conditions?
   ```python
   # Ablate and test on different trial types
   for trial_type in ['congruent', 'incongruent']:
       effect = measure_ablation_effect(neuron_idx, trial_type)
   ```

3. **Recovery**: Can different architectures achieve the same ablated behavior?
   ```python
   # Try recovery with different model architectures
   recovery_configs = [
       {'branching_factors': [5, 5]},
       {'branching_factors': [10, 10]},
       {'recurrence_hidden_layers': [64, 64]},
   ]
   ```

4. **Multiple ablations**: What happens when ablating multiple neurons?
   ```python
   for num_ablated in [1, 2, 5, 10]:
       ablate_neurons_random(num_ablated)
       measure_effect()
   ```

## Troubleshooting

### "Model has no attribute generate_samples"
Make sure you're loading the correct model type. Multi-epoch models should have this method.

### "Ablation has no effect"
- Check that neuron_idx is within the correct range
- Verify the neuron is actually active (check pre-ablation activity)
- Try ablating different neurons or multiple neurons

### "Recovery model doesn't converge"
- Check that ablated distribution is stable (run multiple ablation samplings)
- Verify task inputs are generated correctly
- Reduce learning rate or adjust training schedule

### "Out of memory during sample generation"
- Reduce `num_samples_per_trial` or `num_trials`
- Generate samples in smaller batches
- Clear cache between trials: `torch.cuda.empty_cache()`

## Advanced Topics

### Gradual Ablation

Instead of complete ablation (weight = 0), try gradual scaling:

```python
for scale in [1.0, 0.75, 0.5, 0.25, 0.0]:
    layer.weight.data[:, neuron_idx] *= scale
    measure_effect(scale)
```

### Dynamic Ablation

Ablate only during specific timesteps:

```python
def timestep_ablation_hook(module, input, output, active_timesteps):
    if current_timestep in active_timesteps:
        output[..., neuron_idx] = 0.0
    return output
```

### Population Ablation

Ablate functional populations rather than individual neurons:

```python
# Find neurons with similar selectivity
correlation_matrix = compute_neural_correlations(activations)
population = find_similar_neurons(neuron_idx, correlation_matrix, threshold=0.8)

for n in population:
    ablate_neuron(n)
```

## References

- Your dynamical systems code: `ddpm/dynamical/moment_matching_scheduled.py`
- Model architectures: `ddpm/model/transition/`
- Training loops: `ddpm/train/multiepoch.py`
- Analysis examples: `ddpm/analysis/new_analysis/`

## Questions?

This workflow builds on your existing codebase. Key files to reference:
- Model loading: See `ddpm/analysis/new_analysis/fitting_bns.ipynb`
- Sample generation: See `ddpm/train/multiepoch.py` validation loops
- Moment matching: See `ddpm/dynamical/moment_matching_scheduled.py`
