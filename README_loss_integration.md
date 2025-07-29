# Generalized Flow Matching Loss for Discrete Flow Language Models

This repository contains an adaptation of the generalized loss function from Facebook Research's flow matching repository, specifically designed for discrete flow language models (DFLMs).

## Overview

The generalized flow matching loss provides a more principled approach to training discrete flow models by incorporating the probability path interpolation between source (noise) and target (data) distributions. This is particularly important for discrete flow matching where we need to handle the interpolation in token space.

## Key Components

### 1. `GeneralizedKL` Class

The core generalized KL divergence loss that accounts for probability path interpolation:

```python
loss_fn = GeneralizedKL(
    alpha_t=alpha_t,      # Data mixing coefficient
    sigma_t=sigma_t,      # Noise mixing coefficient  
    d_alpha_t=d_alpha_t,  # Alpha derivative
    d_sigma_t=d_sigma_t,  # Sigma derivative
    reduction="mean"
)
```

**Key Features:**
- Combines standard cross-entropy with flow matching correction terms
- Accounts for the velocity field through derivative terms
- Handles discrete token distributions with proper masking

### 2. `MixturePathGeneralizedKL` Class

Extends the generalized loss to support mixture of probability paths:

```python
loss_fn = MixturePathGeneralizedKL(
    alpha_t=alpha_coeffs,     # [num_paths] or [batch_size, num_paths]
    sigma_t=sigma_coeffs,     # [num_paths] or [batch_size, num_paths]
    d_alpha_t=d_alpha_coeffs, # Derivatives
    d_sigma_t=d_sigma_coeffs, # Derivatives
    mixture_weights=weights,  # Optional mixing weights
    reduction="mean"
)
```

**Benefits:**
- Allows different interpolation schedules within the same batch
- More flexible probability path modeling
- Better handling of diverse text patterns

### 3. `MixtureDiscreteProbPath` Class

Specialized for discrete probability path interpolation:

```python
loss_fn = MixtureDiscreteProbPath(
    vocab_size=tokenizer.vocab_size,
    mask_token_id=tokenizer.mask_token_id,
    time_scheduler=scheduler_function,
    reduction="mean"
)
```

**Features:**
- Direct interpolation between noise and data distributions
- KL divergence between predicted and interpolated distributions
- Proper handling of mask tokens and vocabulary constraints

## Time Schedulers

The library provides three types of schedulers for probability path interpolation:

### Linear Scheduler (Default)
```python
α(t) = t, σ(t) = 1-t
```
Simple linear interpolation between noise and data.

### Cosine Scheduler  
```python
α(t) = 0.5 * (1 + cos(π(1-t)))
σ(t) = 0.5 * (1 + cos(πt))
```
Smoother interpolation with slower changes at the boundaries.

### Exponential Scheduler
```python
α(t) = exp(-2(1-t))
σ(t) = exp(-2t)  
```
More aggressive interpolation with faster convergence.

## Usage Example

### Basic Usage

```python
from loss import MixturePathGeneralizedKL, create_flow_matching_schedulers

# Create schedulers
alpha_t, sigma_t, d_alpha_t, d_sigma_t = create_flow_matching_schedulers(
    num_timesteps=512, 
    device=device, 
    scheduler_type="linear"
)

# Create loss function
loss_fn = MixturePathGeneralizedKL(
    alpha_t=alpha_t,
    sigma_t=sigma_t,
    d_alpha_t=d_alpha_t,
    d_sigma_t=d_sigma_t,
    reduction="mean"
)

# In training loop
logits = model(noised_input_ids, attention_mask, time_t)
loss = loss_fn(
    input=logits.transpose(-1, -2),  # [B, V, L] -> [B, L, V]
    target=target_ids
)
```

### Advanced Usage with Different Schedulers

```python
# Try different schedulers
for scheduler_type in ["linear", "cosine", "exponential"]:
    alpha_t, sigma_t, d_alpha_t, d_sigma_t = create_flow_matching_schedulers(
        num_timesteps=512, 
        device=device, 
        scheduler_type=scheduler_type
    )
    
    loss_fn = MixturePathGeneralizedKL(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        d_alpha_t=d_alpha_t,
        d_sigma_t=d_sigma_t,
        reduction="mean"
    )
    
    # Train model with this scheduler...
```

## Integration with Your Training Loop

The updated training script (`train_updated.py`) shows how to integrate the generalized loss:

### Key Changes from Original Training:

1. **Loss Function Creation:**
```python
# Create flow matching schedulers
alpha_t, sigma_t, d_alpha_t, d_sigma_t = create_flow_matching_schedulers(
    num_timesteps, accelerator.device, scheduler_type
)

# Create the generalized loss function
loss_fn = MixturePathGeneralizedKL(
    alpha_t=alpha_t,
    sigma_t=sigma_t,
    d_alpha_t=d_alpha_t,
    d_sigma_t=d_sigma_t,
    reduction="mean"
)
```

2. **Time-Dependent Noise Scheduling:**
```python
# Sample timesteps for this batch
timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=device)
time_t = alpha_t[timesteps]  # Use alpha_t values as time embeddings

# Create noised inputs using the time-dependent noise schedule
noise_probs = sigma_t[timesteps].unsqueeze(1)  # [B, 1]
noised_input_ids = model._create_noise_ids(input_ids, noise_probs)
```

3. **Loss Computation:**
```python
# Compute loss using the generalized flow matching loss
loss = loss_fn(
    input=logits.transpose(-1, -2),  # [B, V, L] -> [B, L, V]
    target=target,
)
```

## Mathematical Foundation

The generalized loss combines several terms:

1. **Cross-Entropy Term:** Standard token prediction loss
2. **Flow Matching Correction:** Accounts for probability path interpolation
3. **Velocity Field Terms:** Incorporates time derivatives for proper flow dynamics

The loss function implements:
```
L = CE(logits, target) + flow_correction
```

Where:
```
flow_correction = d_α(t) * P_data + d_σ(t) * P_noise - (α(t) * P_data + σ(t) * P_noise)
```

This ensures that the model learns not just to predict tokens, but to follow the correct probability flow from noise to data.

## Benefits of Generalized Flow Matching Loss

1. **Better Training Dynamics:** More stable gradients through proper flow field modeling
2. **Improved Sample Quality:** Better interpolation between noise and data distributions  
3. **Flexible Scheduling:** Multiple scheduler types for different training regimes
4. **Theoretical Soundness:** Grounded in flow matching theory for discrete spaces

## Hyperparameter Recommendations

- **Scheduler Type:** Start with "linear", try "cosine" for smoother training
- **Number of Timesteps:** 256-512 works well for most applications
- **Learning Rate:** Slightly lower than standard CE training (1e-4 instead of 1e-3)
- **Batch Size:** Similar to original training, 32-128 depending on GPU memory

## Monitoring Training

The updated training script includes additional logging:
- Flow matching loss components
- Time step distributions
- Scheduler-specific metrics

Watch for:
- Stable loss convergence (should be smoother than pure CE)
- Reasonable gradient norms (flow matching can help with gradient stability)
- Consistent performance across different time steps

## Troubleshooting

**High Loss Values:** Check that α(t) + σ(t) ≈ 1 for proper normalization

**Unstable Training:** Try cosine scheduler or reduce learning rate

**Poor Generation Quality:** Ensure time embeddings are properly passed to model

**Memory Issues:** Reduce num_timesteps or use gradient checkpointing

## Citation

If you use this implementation, please cite both the original flow matching work and the discrete flow matching papers that inspired this adaptation.

This implementation adapts concepts from:
- Flow Matching for Generative Modeling (Lipman et al.)  
- Discrete Flow Matching (Gat et al.)
- The Facebook Research flow_matching repository