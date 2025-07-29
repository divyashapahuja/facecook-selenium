# Integration Guide: Flow Matching Loss for Discrete Diffusion

This guide shows exactly how to modify your existing training code to use the adapted Flow Matching loss.

## Quick Integration (Minimal Changes)

If you want to make minimal changes to your existing code, use the `DiscreteDiffusionLoss`:

### Step 1: Import the loss
```python
from loss import DiscreteDiffusionLoss
```

### Step 2: Initialize the loss (before your training loop)
```python
# Replace this with your actual mask token ID
mask_token_id = model.mask_token_id if hasattr(model, 'mask_token_id') else 0
loss_fn = DiscreteDiffusionLoss(mask_token_id=mask_token_id, reduction="mean")
```

### Step 3: Replace your loss calculation
Replace this part of your training loop:
```python
# OLD CODE:
loss = F.cross_entropy(
    input=logits.transpose(-1, -2),
    target=target,
    reduction="mean",
)
```

With:
```python
# NEW CODE:
loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
```

That's it! The rest of your training code remains exactly the same.

## Full Flow Matching Integration

For the complete Flow Matching approach, you'll need more changes:

### Step 1: Import required components
```python
from loss import (
    MixturePathGeneralizedKL, 
    MixtureDiscreteProbPath, 
    LinearScheduler
)
```

### Step 2: Initialize the Flow Matching components
```python
# Initialize scheduler and path
scheduler = LinearScheduler()  # or PolynomialConvexScheduler(n=1.0)
path = MixtureDiscreteProbPath(scheduler)
flow_loss = MixturePathGeneralizedKL(path, reduction="mean")
```

### Step 3: Modify your training step
Replace your entire training step logic with:

```python
# In your training loop:
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Sample timesteps and normalize to [0, 1] for Flow Matching
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = timesteps.float() / num_timesteps  # This is the key change!
    
    # Create noise (mask tokens) - this is your x_0
    mask_token_id = model.mask_token_id if hasattr(model, 'mask_token_id') else 0
    x_0 = torch.full_like(input_ids, mask_token_id)
    
    # Sample from the Flow Matching path
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_t)
    noised_input_ids = path_sample.x_t
    
    # Forward pass (same as before)
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Compute Flow Matching loss
    loss = flow_loss(
        logits=logits,
        x_1=input_ids,  # clean tokens
        x_t=noised_input_ids,  # noised tokens
        t=time_t,
        attention_mask=attention_mask
    )
    
    # The rest of your training loop remains the same
    accelerator.backward(loss)
    # ... optimizer step, etc.
```

## Key Differences Between Approaches

| Aspect | Original Code | DiscreteDiffusionLoss | Flow Matching |
|--------|---------------|----------------------|---------------|
| **Changes Required** | - | Minimal (just loss function) | Moderate (sampling logic) |
| **Time Normalization** | `time_scheduler[timesteps]` | Same as original | `timesteps.float() / num_timesteps` |
| **Noise Creation** | `model._create_noise_ids()` | Same as original | `path.sample()` |
| **Loss Calculation** | `F.cross_entropy()` | `loss_fn(logits, target, mask)` | `loss_fn(logits, x_1, x_t, t, mask)` |
| **Theoretical Foundation** | Standard CE | Improved CE with masking | Flow Matching theory |

## Recommended Approach

1. **Start with DiscreteDiffusionLoss**: Make minimal changes and see if you get improvements
2. **If satisfied**: Stop here, you're done!
3. **If you want more**: Try the full Flow Matching approach for potentially better gradients and training dynamics

## Expected Benefits

### DiscreteDiffusionLoss
- Better handling of attention masks
- More robust loss computation
- Minimal code changes

### Full Flow Matching
- Theoretically grounded approach
- Potentially better gradient flow
- State-of-the-art generative modeling framework
- Better handling of the noise-to-data transition

## Troubleshooting

### Common Issues

1. **Shape mismatches**: Make sure your model's output shape is `(batch, seq_len, vocab_size)`
2. **Time normalization**: Flow Matching expects times in `[0, 1]`, your original code might use different ranges
3. **Mask token ID**: Make sure you're using the correct mask token ID for your tokenizer

### Debug Tips

1. **Print shapes**: Add debug prints to check tensor shapes at each step
2. **Compare losses**: Run both old and new loss side-by-side to compare values
3. **Start small**: Test with a small batch first before full training

## Performance Notes

- The `DiscreteDiffusionLoss` should have similar performance to your original code
- The full Flow Matching approach might be slightly slower due to path sampling, but should provide better training dynamics
- Both approaches are fully compatible with your existing accelerator setup