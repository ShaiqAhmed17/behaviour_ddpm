# Model-to-Model Recovery: Understanding the Training Setup

## The Critical Question

**Q: Don't I need a different config since I'm training on different outputs?**

**A: No, but you need to understand what the config controls!**

## What the Config Actually Controls

Your config file specifies two separate things:

### 1. Task Structure (Inputs)
```yaml
task_name: indexing_cue_first_fixed_probability_vectoral
task_config:
  num_items: 2
  stimulus_exposure_duration: 5
  pre_index_delay_duration: [15, 15]
  # etc.
```

This defines:
- What stimuli are presented
- Timing of task epochs
- Cue structure
- Input dimensionality

**This stays the same in recovery training!** You're still doing the same task, just matching different behavior.

### 2. Model Architecture
```yaml
model_name: hvae_delayed_index_cue_dendritic
model_config:
  sample_ambient_dim: 16
  branching_factors: [10, 10]
  # etc.
```

This defines:
- Network architecture
- Number of neurons
- Diffusion schedule

**This can also stay the same** (though you could try different architectures to see if they can achieve the same behavior).

## What Changes in Recovery Training

The key difference is **where the target behavior comes from**:

### Normal Training
```python
# Generate everything from task
task_vars = task.task_variable_gen.generate()
sensory = task.sensory_gen.generate(**task_vars)
samples = task.sample_gen.generate(**task_vars)  # ← Target behavior from task

# Train to match task-generated behavior
model.train_on(sensory_input=sensory, target_samples=samples)
```

### Recovery Training
```python
# Inputs from task, targets from ablated model
task_vars = task.task_variable_gen.generate()
sensory = task.sensory_gen.generate(**task_vars)
samples = ablated_samples[random_indices]  # ← Target behavior from ablated model!

# Train to match ablated behavior
model.train_on(sensory_input=sensory, target_samples=samples)
```

## The Complete Picture

Here's what happens at each training step:

```python
# Step 1: Get task inputs (from config/task)
task_vars = task.task_variable_gen.generate()
# → e.g., which items to show, when to cue, etc.

sensory = task.sensory_gen.generate(**task_vars)
# → e.g., stimulus features, cue vectors, etc.

# Step 2: Get target behavior (from ablated samples, NOT task!)
target_batch = ablated_samples[random_batch]
# → This is the key difference!

# Step 3: Add noise to targets (forward diffusion)
noise_dict = model.noise(target_batch, sensory_input=sensory)
x_t = noise_dict['x_t']  # Noised targets

# Step 4: Train model to remove noise (reverse diffusion)
pred_dict = model.residual(x_t, sensory_input=sensory)
loss = MSE(pred_dict['epsilon_hat'], noise_dict['epsilon'])
loss.backward()
```

## Why This Works

The model learns:
- **From sensory inputs**: "Given this task condition..."
- **To target samples**: "...produce this behavior"

In recovery training:
- **Same task conditions** (same cues, same stimuli)
- **Different target behavior** (ablated distribution)

So the model learns: "Given the same task inputs, produce the ablated behavior instead of the original behavior"

## Practical Example

Let's say your task is delayed index cuing with 2 items:

### Normal Training
```
Input: [item1=30°, item2=150°, cue=index_1]
Target: Sample near 30° (from task.sample_gen)
Model learns: "Cue to first item → produce samples around first item"
```

### After Ablating Neuron 5
```
Input: [item1=30°, item2=150°, cue=index_1]  (same!)
Output: Samples now biased (e.g., shifted +10°, more variable)
```

### Recovery Training
```
Input: [item1=30°, item2=150°, cue=index_1]  (still same!)
Target: Shifted/biased samples (from ablated_samples)
Model learns: "Cue to first item → produce shifted/biased samples"
```

## Do You Ever Need a Different Config?

You might want to change the config if:

### 1. Testing Different Architectures
```yaml
# Original
branching_factors: [10, 10]

# Recovery attempt with simpler architecture
branching_factors: [5, 5]
```

**Question**: Can a simpler network achieve the ablated behavior?

### 2. Adjusting Diffusion Schedule
```yaml
# Original
ultimate_sigma2: 0.30
starting_sigma2: 0.01

# Recovery with less noise
ultimate_sigma2: 0.15  # Less stochastic
starting_sigma2: 0.01
```

**Question**: Does the ablated behavior require less/more stochasticity?

### 3. Changing Training Dynamics
```yaml
# Original
num_trials: 10_000_000
lr: 0.001

# Recovery with more careful training
num_trials: 20_000_000  # Train longer
lr: 0.0005  # More stable
```

But these are **optional experiments**, not required!

## Common Misconception

❌ **Wrong thinking**: "I need to modify task.sample_gen to generate ablated samples"

✅ **Correct thinking**: "I keep task.sample_gen as-is (for validation), but train the model using ablated samples as targets"

## Code Implementation

Here's how the recovery training script handles this:

```python
def train_recovery_model(model, task, target_samples, ...):
    """
    Args:
        task: Used for INPUT generation only
        target_samples: Used for TARGET generation (ablated samples)
    """
    for epoch in range(num_epochs):
        # Get INPUTS from task
        task_vars = task.task_variable_gen.generate()
        sensory = task.sensory_gen.generate(**task_vars)
        
        # Get TARGETS from ablated samples
        indices = torch.randint(0, len(target_samples), (batch_size,))
        targets = target_samples[indices]  # ← NOT from task!
        
        # Train on (input, target) pairs
        noise_dict = model.noise(targets, sensory_input=sensory)
        pred_dict = model.residual(noise_dict['x_t'], sensory)
        loss = mse_loss(pred_dict['pred'], noise_dict['target'])
        loss.backward()
```

## Summary

**You don't need a different config** because:
1. Task structure (inputs) stays the same
2. Model architecture can stay the same (or be changed for experiments)
3. The training loop handles using ablated samples as targets

**The key modification** is in the training loop, not the config:
- Replace `task.sample_gen.generate()` with `ablated_samples[indices]`
- Everything else stays the same

This is already implemented correctly in `recovery_training.py`!

## Validation Note

After recovery training, you can compare:

```python
# Generate from recovery model
recovery_samples = recovery_model.generate_samples(task_input)

# Compare to ablated target
ablated_samples = torch.load('ablated_samples.pt')

# They should match!
distance = torch.norm(recovery_samples.mean() - ablated_samples.mean())
print(f"Recovery accuracy: {distance:.4f}")
```

The recovery model has learned to produce the ablated behavior given the same task inputs!
