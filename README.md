# Flow Matching Loss for Discrete Diffusion Training

This repository contains an adapted version of the Facebook Research Flow Matching generalized loss module, specifically adjusted for discrete diffusion model training.

## Overview

The original Flow Matching loss from [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) has been adapted to work with discrete diffusion models for text generation. This adaptation provides two main approaches:

1. **MixturePathGeneralizedKL**: A faithful adaptation of the Flow Matching generalized KL loss
2. **DiscreteDiffusionLoss**: A simplified loss that's closer to standard cross-entropy but with better handling of masked tokens

## Key Components

### Schedulers
- `LinearScheduler`: Linear interpolation between noise and data
- `PolynomialConvexScheduler`: Polynomial interpolation with configurable power

### Path Sampling
- `MixtureDiscreteProbPath`: Defines how to sample along the probability path from noise to data

### Loss Functions
- `MixturePathGeneralizedKL`: Flow Matching loss adapted for discrete tokens
- `DiscreteDiffusionLoss`: Simplified cross-entropy based loss with proper masking

## Integration with Your Training Code

### Option 1: Flow Matching Loss

```python
from loss import MixturePathGeneralizedKL, MixtureDiscreteProbPath, LinearScheduler

# Setup
scheduler = LinearScheduler()  # or PolynomialConvexScheduler(n=1.0)
path = MixtureDiscreteProbPath(scheduler)
loss_fn = MixturePathGeneralizedKL(path, reduction="mean")

# In your training loop
def training_step(model, batch, num_timesteps, accelerator):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Sample timesteps and normalize to [0, 1]
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = timesteps.float() / num_timesteps
    
    # Create noise (mask tokens)
    x_0 = torch.full_like(input_ids, model.mask_token_id)
    
    # Sample from path
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_t)
    noised_input_ids = path_sample.x_t
    
    # Forward pass
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Compute loss
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,  # clean tokens
        x_t=noised_input_ids,  # noised tokens
        t=time_t,
        attention_mask=attention_mask
    )
    
    return loss
```

### Option 2: Simplified Discrete Diffusion Loss

```python
from loss import DiscreteDiffusionLoss, create_time_scheduler

# Setup
loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")
time_scheduler = create_time_scheduler(num_timesteps=512, scheduler_type="linear", device=accelerator.device)

# In your training loop (minimal changes from your original code)
def training_step(model, batch, num_timesteps, accelerator):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Sample timesteps (same as original)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    # Create noised inputs (same as original)
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    # Forward pass
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Prepare target (same as original)
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # Compute loss with new loss function
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    return loss
```

## Key Differences from Original Flow Matching

1. **Discrete Tokens**: Adapted for discrete token spaces instead of continuous spaces
2. **Attention Masking**: Proper handling of attention masks for sequence models
3. **Mask Token Integration**: Works with your existing mask token approach
4. **Training Loop Compatibility**: Minimal changes required to your existing training code

## Benefits

1. **Theoretical Foundation**: Based on the solid mathematical foundation of Flow Matching
2. **Flexibility**: Two approaches depending on how much you want to change your existing code
3. **Better Gradients**: Flow Matching can provide better gradient flow compared to standard cross-entropy
4. **State-of-the-art**: Based on recent advances in generative modeling

## Usage Examples

See `training_example.py` for complete examples of how to integrate both loss approaches with your training code.

### Running the Example

```python
python training_example.py
```

This will show a comparison between the original cross-entropy loss, the new discrete diffusion loss, and the Flow Matching loss.

## Requirements

- PyTorch
- Your existing model and training infrastructure

## License

This code is adapted from Facebook Research's Flow Matching repository and maintains the same CC-by-NC license.
