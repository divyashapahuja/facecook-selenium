# Integration Guide: Replacing Cross-Entropy with Generalized Loss

This guide explains how to replace the cross-entropy loss in your discrete flow matching model with the generalized KL loss from Facebook Research's flow matching repository.

## Overview

The `MixturePathGeneralizedKL` loss is specifically designed for discrete flow matching and provides better theoretical grounding than standard cross-entropy loss for this application.

## Key Changes Required

### 1. Import the Generalized Loss

```python
from generalized_loss import MixturePathGeneralizedKL

# Initialize the loss function
generalized_loss_fn = MixturePathGeneralizedKL(reduction="mean")
```

### 2. Implement Scheduler Parameter Computation

You need to implement the `compute_scheduler_params` function based on your specific time scheduler:

```python
def compute_scheduler_params(time_t, num_timesteps):
    """
    Compute alpha_t and d_alpha_t based on your time scheduler.
    
    Args:
        time_t: Current time values from your scheduler
        num_timesteps: Total number of timesteps
    
    Returns:
        alpha_t: Scheduler parameter alpha at time t
        d_alpha_t: Derivative of alpha_t with respect to time
    """
    # IMPORTANT: Replace this with your actual scheduler logic
    # This is just an example for a linear schedule
    alpha_t = 1.0 - time_t  
    d_alpha_t = -torch.ones_like(time_t)
    
    return alpha_t, d_alpha_t
```

### 3. Replace Cross-Entropy Loss Calls

#### Before (Cross-Entropy):
```python
loss = F.cross_entropy(
    input=logits.transpose(-1, -2),
    target=target,
    reduction="mean",
)
```

#### After (Generalized Loss):
```python
# Compute scheduler parameters
alpha_t, d_alpha_t = compute_scheduler_params(time_t, num_timesteps)

# Use generalized loss
loss = generalized_loss_fn(
    logits=logits,
    x_1=target,  # Original clean data
    x_t=noised_input_ids,  # Noised data
    t=time_t,
    alpha_t=alpha_t,
    d_alpha_t=d_alpha_t
)
```

### 4. Handle Masking

The generalized loss doesn't use `-100` masking like cross-entropy. Instead, you have two options:

#### Option A: Apply mask after loss computation
```python
# Don't set target to -100, keep original values
target = input_ids.clone()

# Compute loss
loss = generalized_loss_fn(...)

# Apply attention mask
valid_mask = attention_mask.unsqueeze(-1).expand_as(target)
loss = loss * valid_mask.float()
loss = loss.sum() / valid_mask.sum()
```

#### Option B: Filter valid positions before loss computation
```python
# Create masks for valid positions
masked_positions = noised_input_ids != model.mask_token_id
padding_positions = attention_mask == 0
valid_positions = ~masked_positions & ~padding_positions

if valid_positions.sum() > 0:
    # Extract only valid positions
    valid_logits = logits[valid_positions]
    valid_targets = target[valid_positions]
    # ... compute loss on valid positions only
```

## Complete Integration Example

Here's how to modify your existing functions:

### Modified evaluate_model function:

```python
def evaluate_model(model, val_loader, num_timesteps, time_scheduler, accelerator, max_eval_batches=50):
    """Evaluate model on validation set using generalized loss"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_eval_batches:
                break
                
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]
            
            noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
            logits = model(noised_input_ids, attention_mask, time_t)
            
            # Prepare target (no -100 masking)
            target = input_ids.clone()
            
            # Compute scheduler parameters
            alpha_t, d_alpha_t = compute_scheduler_params(time_t, num_timesteps)
            
            # Use generalized loss
            loss = generalized_loss_fn(
                logits=logits,
                x_1=target,
                x_t=noised_input_ids,
                t=time_t,
                alpha_t=alpha_t,
                d_alpha_t=d_alpha_t
            )
            
            # Apply attention mask
            valid_mask = attention_mask.unsqueeze(-1).expand_as(target)
            loss = loss * valid_mask.float()
            loss = loss.sum() / valid_mask.sum()
            
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    return total_loss / max(total_batches, 1)
```

### Modified training loop:

```python
# In your training loop, replace the cross-entropy section with:

# Prepare target (no -100 masking)
target = input_ids.clone()

# Compute scheduler parameters
alpha_t, d_alpha_t = compute_scheduler_params(time_t, num_timesteps)

# Use generalized loss instead of cross-entropy
loss = generalized_loss_fn(
    logits=logits,
    x_1=target,
    x_t=noised_input_ids,
    t=time_t,
    alpha_t=alpha_t,
    d_alpha_t=d_alpha_t
)

# Apply attention mask
valid_mask = attention_mask.unsqueeze(-1).expand_as(target)
loss = loss * valid_mask.float()
loss = loss.sum() / valid_mask.sum()

# Continue with backward pass
accelerator.backward(loss)
```

## Important Notes

1. **Scheduler Parameters**: The most critical part is implementing `compute_scheduler_params` correctly based on your specific time scheduler. The `alpha_t` and `d_alpha_t` parameters must match your scheduler's definition.

2. **Shape Compatibility**: The generalized loss expects:
   - `logits`: (batch, seq_len, vocab_size)
   - `x_1`, `x_t`: (batch, seq_len)
   - `t`, `alpha_t`, `d_alpha_t`: (batch,)

3. **Masking Strategy**: Choose between post-loss masking (Option A) or pre-loss filtering (Option B) based on your computational preferences and memory constraints.

4. **Debugging**: Start with small batch sizes and verify that the loss values are reasonable compared to your previous cross-entropy loss values.

## Testing the Integration

1. Run a few training steps with both loss functions and compare:
   - Loss magnitudes should be in similar ranges
   - Gradients should be reasonable (not exploding/vanishing)
   - Memory usage should be comparable

2. Monitor training metrics:
   - Training loss should decrease over time
   - Validation metrics should improve
   - Model outputs should remain coherent

## Troubleshooting

- **NaN losses**: Check that `alpha_t` never equals 1.0 (division by zero in `d_alpha_t / (1 - alpha_t)`)
- **Shape errors**: Verify tensor shapes match the expected dimensions
- **Memory issues**: Consider using gradient checkpointing or smaller batch sizes if needed