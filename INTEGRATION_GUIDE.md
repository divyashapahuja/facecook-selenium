# Integration Guide: Flow Matching Loss for Discrete Diffusion

This guide shows exactly how to modify your existing training code to use the adapted Flow Matching loss.

## Option 1: Use Your Existing time_scheduler (Recommended)

This approach requires **zero changes** to your time scheduling logic:

### Step 1: Import the helper function
```python
from loss import training_step_with_existing_scheduler
```

### Step 2: Replace your training step
In your training loop, replace the loss calculation with:

```python
# Your existing setup (no changes needed!)
time_scheduler = torch.linspace(
    1 / num_timesteps,
    1,
    steps=num_timesteps,
    dtype=torch.float32,
    device=accelerator.device,
)

# In your training loop:
for batch in train_loader:
    # OLD CODE (replace this entire block):
    # input_ids = batch["input_ids"]
    # attention_mask = batch["attention_mask"].bool()
    # timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    # time_t = time_scheduler[timesteps]
    # noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    # logits = model(noised_input_ids, attention_mask, time_t)
    # target = input_ids.clone()
    # target[noised_input_ids != model.mask_token_id] = -100
    # target[attention_mask == 0] = -100
    # loss = F.cross_entropy(input=logits.transpose(-1, -2), target=target, reduction="mean")
    
    # NEW CODE (single function call):
    loss, logits, noised_input_ids = training_step_with_existing_scheduler(
        model=model,
        batch=batch,
        num_timesteps=num_timesteps,
        time_scheduler=time_scheduler,  # Your existing scheduler!
        accelerator=accelerator,
        loss_type="discrete"  # or "flow_matching" for full Flow Matching
    )
    
    # Rest of your training loop stays the same
    accelerator.backward(loss)
    # ... optimizer step, etc.
```

### Step 3: Choose your loss type
```python
# For minimal changes (improved cross-entropy):
loss_type="discrete"

# For full Flow Matching benefits:
loss_type="flow_matching"
```

## Option 2: Manual Integration with time_scheduler

If you prefer more control:

### For Discrete Diffusion Loss:
```python
from loss import DiscreteDiffusionLoss

# Initialize once
loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")

# In your training loop (minimal changes):
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Your existing time sampling (no changes)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    # Your existing noise creation (no changes)
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    # Forward pass (no changes)
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Your existing target preparation (no changes)
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # ONLY CHANGE: Replace F.cross_entropy with new loss
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
```

### For Flow Matching Loss:
```python
from loss import create_flow_matching_loss_with_scheduler

# Initialize once
path, flow_loss_fn = create_flow_matching_loss_with_scheduler(
    mask_token_id=model.mask_token_id,
    scheduler_type="linear"
)

# In your training loop:
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Your existing time sampling (no changes)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    # Flow Matching path sampling (replaces model._create_noise_ids)
    x_0 = torch.full_like(input_ids, model.mask_token_id)
    time_normalized = (time_t - time_t.min()) / (time_t.max() - time_t.min())
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_normalized)
    noised_input_ids = path_sample.x_t
    
    # Forward pass (no changes)
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Flow Matching loss (handles time normalization automatically)
    loss = flow_loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        t=time_t,  # Your original time_scheduler values!
        attention_mask=attention_mask
    )
```

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

## Why Use Your Existing time_scheduler?

1. **Zero Changes**: Your time sampling logic stays exactly the same
2. **Consistent**: Same time values go to your model
3. **Compatible**: Works with your existing `model._create_noise_ids()` method
4. **Flexible**: Easy to switch between loss types

## Key Differences Between Approaches

| Aspect | Original Code | With time_scheduler | Without time_scheduler |
|--------|---------------|---------------------|----------------------|
| **Time Sampling** | `time_scheduler[timesteps]` | Same | `timesteps.float() / num_timesteps` |
| **Noise Creation** | `model._create_noise_ids()` | Same | `path.sample()` |
| **Model Input** | `time_scheduler` values | Same | Normalized `[0,1]` values |
| **Changes Required** | - | Minimal | Moderate |

## Recommended Approach

1. **Start with Option 1**: Use `training_step_with_existing_scheduler()` with `loss_type="discrete"`
2. **Test**: Compare training metrics with your original setup
3. **Upgrade**: Try `loss_type="flow_matching"` if you want full Flow Matching benefits
4. **Optimize**: Fine-tune based on your results

This approach lets you get the benefits of the improved loss functions while keeping your existing, working time scheduling logic intact!