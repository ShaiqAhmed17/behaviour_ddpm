import torch
import numpy as np
from typing import Tuple

def log_normal_to_discrete(log_means, log_vars, dt, max_timesteps, method='midpoint'):
    """
    Convert log-normal distribution parameters to discrete response time distributions.
    
    Args:
        log_means: [batch, trials] - means of log-normal distributions (in log space)
        log_vars: [batch, trials] - variances of log-normal distributions (in log space)
        dt: float - discretization time step
        max_timesteps: int - maximum number of timesteps
        method: str - 'midpoint' or 'integrate' for discretization method
        
    Returns:
        discrete_probs: [batch, trials, max_timesteps] - discrete probability distribution
    """
    batch_size, num_trials = log_means.shape
    device = log_means.device
    
    # Create time grid
    if method == 'midpoint':
        # Evaluate PDF at midpoint of each interval
        times = torch.arange(max_timesteps, device=device, dtype=torch.float32) * dt + dt/2
        times = times.unsqueeze(0).unsqueeze(0).expand(batch_size, num_trials, -1)  # [batch, trials, max_timesteps]
        
        # Log-normal PDF: f(x) = 1/(x*σ*√(2π)) * exp(-((ln(x) - μ)²)/(2σ²))
        log_stds = torch.sqrt(log_vars).unsqueeze(2)  # [batch, trials, 1]
        log_means_expanded = log_means.unsqueeze(2)   # [batch, trials, 1]
        
        # Compute log PDF for numerical stability
        log_times = torch.log(times + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Log PDF = -log(x) - log(σ) - 0.5*log(2π) - ((ln(x) - μ)²)/(2σ²)
        log_pdf = (-log_times 
                   - torch.log(log_stds) 
                   - 0.5 * np.log(2 * np.pi)
                   - ((log_times - log_means_expanded) ** 2) / (2 * log_vars.unsqueeze(2)))
        
        # Convert to probabilities and multiply by dt for discrete approximation
        probs = torch.exp(log_pdf) * dt
        
    elif method == 'integrate':
        # Integrate over each interval using CDF differences
        times_start = torch.arange(max_timesteps, device=device, dtype=torch.float32) * dt
        times_end = times_start + dt
        
        # Expand for batch and trial processing
        times_start = times_start.unsqueeze(0).unsqueeze(0).expand(batch_size, num_trials, -1)
        times_end = times_end.unsqueeze(0).unsqueeze(0).expand(batch_size, num_trials, -1)
        
        # Convert to standard normal for CDF computation
        log_stds = torch.sqrt(log_vars).unsqueeze(2)
        log_means_expanded = log_means.unsqueeze(2)
        
        z_start = (torch.log(times_start + 1e-8) - log_means_expanded) / log_stds
        z_end = (torch.log(times_end + 1e-8) - log_means_expanded) / log_stds
        
        # Use standard normal CDF
        cdf_start = 0.5 * (1 + torch.erf(z_start / np.sqrt(2)))
        cdf_end = 0.5 * (1 + torch.erf(z_end / np.sqrt(2)))
        
        probs = cdf_end - cdf_start
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to ensure probabilities sum to 1
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    return probs


def response_time_loss(hazard_rates, target_log_means, target_log_vars, dt):
    """
    Compute cross-entropy loss between predicted and target response time distributions.
    
    Args:
        hazard_rates: [batch, num_trials, max_timesteps] - predicted hazard rates
        target_log_means: [batch, num_trials] - target log-normal means
        target_log_vars: [batch, num_trials] - target log-normal variances  
        dt: float - discretization time step
        
    Returns:
        loss: [batch, num_trials] - cross-entropy loss for each trial
    """
    batch_size, num_trials, max_timesteps = hazard_rates.shape
    
    # Convert hazard rates to RT distributions
    pred_rt_probs = hazard_to_distribution(hazard_rates)  # [batch, num_trials, max_timesteps]
    
    # Convert target log-normal to discrete distributions
    target_rt_probs = log_normal_to_discrete(
        target_log_means, target_log_vars, dt, max_timesteps
    )  # [batch, num_trials, max_timesteps]
    
    # Compute cross-entropy loss trial-wise
    # CE = -∑ target * log(pred)
    log_pred_probs = torch.log(pred_rt_probs + 1e-8)
    ce_loss = -torch.sum(target_rt_probs * log_pred_probs, dim=-1)  # [batch, num_trials]
    
    return ce_loss


def sample_response_times_batch_trialwise(hazard_rates: torch.Tensor, samples_per_trial: int = 50) -> torch.Tensor:
    """
    Sample response times from hazard rates preserving trial structure.
    
    Args:
        hazard_rates: [batch_size, num_trials, max_timesteps] - hazard rates
        samples_per_trial: int - number of samples to generate per trial
        
    Returns:
        response_times: [batch_size, num_trials, samples_per_trial] - response times per trial
    """
    batch_size, num_trials, max_timesteps = hazard_rates.shape
    device = hazard_rates.device
    
    # Reshape to process all trials together
    hazard_flat = hazard_rates.view(-1, max_timesteps)  # [batch_size * num_trials, max_timesteps]
    total_trials = batch_size * num_trials
    
    # Expand hazard rates for multiple samples per trial
    hazard_expanded = hazard_flat.unsqueeze(1).expand(-1, samples_per_trial, -1)  # [total_trials, samples_per_trial, max_timesteps]
    hazard_expanded = hazard_expanded.reshape(-1, max_timesteps)  # [total_trials * samples_per_trial, max_timesteps]
    
    # Compute survival probabilities
    one_minus_hazard = 1.0 - hazard_expanded
    survival_probs = torch.cumprod(torch.cat([
        torch.ones(total_trials * samples_per_trial, 1, device=device),
        one_minus_hazard[:, :-1]
    ], dim=1), dim=1)  # [total_trials * samples_per_trial, max_timesteps]
    
    # Generate random numbers for inverse CDF sampling
    random_vals = torch.rand(total_trials * samples_per_trial, 1, device=device)
    
    # Find first timestep where survival probability drops below random value
    survived = survival_probs >= random_vals
    response_times = torch.argmax((~survived).float(), dim=1)
    
    # Handle case where all timesteps survive
    never_responded = survived.all(dim=1)
    response_times[never_responded] = max_timesteps - 1
    
    # Reshape back to trial structure: [batch_size, num_trials, samples_per_trial]
    response_times = response_times.view(batch_size, num_trials, samples_per_trial)
    
    return response_times.float()


def fit_lognormal_parameters_by_trial(response_times_trialwise, dt, exclude_max=False, max_timesteps=None):
    """
    Fit log-normal parameters from sampled response times, per trial.
    
    Args:
        response_times_trialwise: [batch_size, num_trials, samples_per_trial] - response times per trial
        dt: float - discretization time step
        exclude_max: bool - if True, exclude response times that hit the maximum timestep
        max_timesteps: int - maximum timesteps (needed if exclude_max=True)
        
    Returns:
        fitted_log_means: [batch_size, num_trials] - fitted μ for each trial
        fitted_log_vars: [batch_size, num_trials] - fitted σ² for each trial
    """
    batch_size, num_trials, samples_per_trial = response_times_trialwise.shape
    device = response_times_trialwise.device
    
    fitted_log_means = torch.zeros(batch_size, num_trials, device=device)
    fitted_log_vars = torch.zeros(batch_size, num_trials, device=device)
    
    for b in range(batch_size):
        for t in range(num_trials):
            trial_times = response_times_trialwise[b, t]
            
            if exclude_max and max_timesteps is not None:
                # Exclude samples that hit maximum timestep
                valid_mask = trial_times < (max_timesteps - 1)
                if valid_mask.sum() > 1:  # Need at least 2 samples for variance
                    trial_times = trial_times[valid_mask]
            
            # Convert timesteps to continuous time
            continuous_times = trial_times * dt
            continuous_times = torch.clamp(continuous_times, min=dt/10)
            
            # Take log of times
            log_times = torch.log(continuous_times)
            
            # Compute sample mean and variance
            fitted_log_means[b, t] = torch.mean(log_times)
            if len(log_times) > 1:
                fitted_log_vars[b, t] = torch.var(log_times, unbiased=True)
            else:
                fitted_log_vars[b, t] = 0.1  # Fallback variance
    
    return fitted_log_means, fitted_log_vars


def fit_lognormal_parameters(response_times_batch, dt, exclude_max=False, max_timesteps=None):
    """Fit log-normal parameters from [batch, samples] response-time tensors.

    This keeps compatibility with older callers that expect batch-level fitting
    rather than per-trial fitting.
    """
    if response_times_batch.ndim != 2:
        raise ValueError(
            f"Expected response_times_batch shape [batch, samples], got {tuple(response_times_batch.shape)}"
        )

    batch_size = response_times_batch.shape[0]
    device = response_times_batch.device
    fitted_log_means = torch.zeros(batch_size, device=device)
    fitted_log_vars = torch.zeros(batch_size, device=device)

    for b in range(batch_size):
        sample_times = response_times_batch[b]

        if exclude_max and max_timesteps is not None:
            valid_mask = sample_times < (max_timesteps - 1)
            if valid_mask.sum() > 1:
                sample_times = sample_times[valid_mask]

        continuous_times = torch.clamp(sample_times * dt, min=dt / 10)
        log_times = torch.log(continuous_times)

        fitted_log_means[b] = torch.mean(log_times)
        fitted_log_vars[b] = torch.var(log_times, unbiased=True) if len(log_times) > 1 else 0.1

    return fitted_log_means, fitted_log_vars


# Keep existing functions that don't need changes
def hazard_to_distribution(hazard_rates, eps=1e-8):
    """Convert discrete hazard rates to response time probability distributions."""
    hazard_rates = torch.clamp(hazard_rates, 0.0, 1.0 - eps)
    batch_size, num_trials, max_timesteps = hazard_rates.shape
    
    # Compute survival probabilities without in-place operations
    one_minus_hazard = 1 - hazard_rates
    # Shift by one timestep for survival calculation
    survival_factors = torch.cat([
        torch.ones(batch_size, num_trials, 1, device=hazard_rates.device),
        one_minus_hazard[:, :, :-1]
    ], dim=2)
    survival_probs = torch.cumprod(survival_factors, dim=2)
    
    # Response time probability at time t = λ_t * P(survived until t)
    rt_probs = hazard_rates * survival_probs
    
    # Normalize to ensure probabilities sum to 1 (handle numerical errors)
    rt_probs = rt_probs / (rt_probs.sum(dim=-1, keepdim=True) + eps)
    
    return rt_probs


def sample_response_times_batch(hazard_rates: torch.Tensor, samples_per_batch: int = 1000) -> torch.Tensor:
    """Sample response times from hazard rates for each batch element (vectorized)."""
    batch_size, num_trials, max_timesteps = hazard_rates.shape
    device = hazard_rates.device
    
    # Flatten to process all batch elements and trials together
    hazard_flat = hazard_rates.view(-1, max_timesteps)  # [batch_size * num_trials, max_timesteps]
    total_sources = batch_size * num_trials
    
    # Determine samples per source
    samples_per_source = max(1, samples_per_batch // total_sources)
    total_samples = total_sources * samples_per_source
    
    # Expand hazard rates for all samples
    hazard_expanded = hazard_flat.unsqueeze(1).expand(-1, samples_per_source, -1)  # [total_sources, samples_per_source, max_timesteps]
    hazard_expanded = hazard_expanded.reshape(-1, max_timesteps)  # [total_samples, max_timesteps]
    
    # Compute cumulative survival probabilities: P(survive until t) = ∏(1 - λ_i) for i < t
    one_minus_hazard = 1.0 - hazard_expanded
    survival_probs = torch.cumprod(torch.cat([
        torch.ones(total_samples, 1, device=device),
        one_minus_hazard[:, :-1]
    ], dim=1), dim=1)  # [total_samples, max_timesteps]
    
    # Generate random numbers for inverse CDF sampling
    random_vals = torch.rand(total_samples, 1, device=device)  # [total_samples, 1]
    
    # Find first timestep where survival probability drops below random value
    survived = survival_probs >= random_vals  # [total_samples, max_timesteps]
    
    # Find the first False (first timestep where we don't survive)
    response_times = torch.argmax((~survived).float(), dim=1)  # [total_samples]
    
    # Handle case where all timesteps survive (no response) - set to max timestep
    never_responded = survived.all(dim=1)
    response_times[never_responded] = max_timesteps - 1
    
    # Reshape back to batch format and subsample if needed
    response_times = response_times.view(total_sources, samples_per_source)  # [total_sources, samples_per_source]
    
    # Group by batch element and subsample to exact samples_per_batch
    all_batch_samples = []
    for b in range(batch_size):
        start_idx = b * num_trials
        end_idx = (b + 1) * num_trials
        batch_samples = response_times[start_idx:end_idx].flatten()  # [num_trials * samples_per_source]
        
        # Subsample to exactly samples_per_batch
        if len(batch_samples) >= samples_per_batch:
            indices = torch.randperm(len(batch_samples), device=device)[:samples_per_batch]
            batch_samples = batch_samples[indices]
        else:
            # Pad with additional samples if needed
            padding_needed = samples_per_batch - len(batch_samples)
            padding = torch.randint(0, max_timesteps, (padding_needed,), device=device)
            batch_samples = torch.cat([batch_samples, padding])
        
        all_batch_samples.append(batch_samples)
    
    return torch.stack(all_batch_samples, dim=0).float()  # [batch_size, samples_per_batch]