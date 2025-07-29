# Flow Matching Loss for Discrete Diffusion - Final Version

This repository contains the final, adapted version of the Facebook Research Flow Matching generalized loss module, specifically designed to work with your existing `time_scheduler` and discrete diffusion training setup.

## 🎯 What's Different

This version **directly uses your existing `time_scheduler`** without any normalization or complex wrapper functions. It's designed to be a drop-in replacement for your current loss calculation.

## 📁 Files

- **`loss_final.py`** - The main loss module with Flow Matching and discrete diffusion losses
- **`training_final.py`** - Complete training script showing integration
- **`README_final.md`** - This documentation

## 🚀 Quick Start

### 1. Import the Loss Functions

```python
from loss_final import MixturePathGeneralizedKL, MixtureDiscreteProbPath, DiscreteDiffusionLoss
```

### 2. Replace Your Loss Calculation

**Option A: Flow Matching Loss (Recommended)**
```python
# Your existing time_scheduler (no changes!)
time_scheduler = torch.linspace(
    1 / num_timesteps,
    1, 
    steps=num_timesteps,
    dtype=torch.float32,
    device=accelerator.device
)

# Create loss components ONCE before training loop
path = MixtureDiscreteProbPath(time_scheduler)
loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")

# In your training loop:
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Your existing timestep sampling (no changes!)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]  # For your model
    
    # Flow Matching approach
    x_0 = torch.full_like(input_ids, model.mask_token_id)
    path_sample = path.sample(x_0=x_0, x_1=input_ids, timesteps=timesteps)
    noised_input_ids = path_sample.x_t
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # NEW: Flow Matching loss
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        timesteps=timesteps,  # Pass timestep indices, not time values!
        attention_mask=attention_mask
    )
    
    # Rest of your training loop unchanged
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

**Option B: Discrete Diffusion Loss (Minimal Changes)**
```python
# Create loss function ONCE
loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")

# In your training loop - only the loss calculation changes:
for batch in train_loader:
    # ... your existing code until loss calculation ...
    
    # Your existing target preparation (unchanged)
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # NEW: Better loss calculation
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    # Rest unchanged
```

## 🔧 Key Changes from Your Original Code

1. **Import**: Add `from loss_final import ...`
2. **Create loss components ONCE** before your training loop (not inside it)
3. **Choose loss type**: Flow Matching or discrete diffusion
4. **Everything else stays the same**: Your time_scheduler, model, data loading, etc.

## 📊 Comparison

| Aspect | Your Original | Flow Matching | Discrete Loss |
|--------|---------------|---------------|---------------|
| **Changes Required** | - | Moderate | Minimal |
| **Time Scheduler** | `time_scheduler[timesteps]` | Same | Same |
| **Model Input** | `time_t` values | Same | Same |
| **Loss Arguments** | `F.cross_entropy(...)` | `loss_fn(logits, x_1, x_t, timesteps, mask)` | `loss_fn(logits, target, mask)` |
| **Theoretical Foundation** | Standard CE | Flow Matching theory | Improved CE |

## 🎛️ Complete Training Script

The `training_final.py` file contains a complete training script that supports both loss types:

```python
# Flow Matching
main_with_flow_matching(
    model, train_dataset, val_dataset,
    loss_type="flow_matching",  # Use Flow Matching
    # ... other parameters
)

# Discrete Diffusion  
main_with_flow_matching(
    model, train_dataset, val_dataset,
    loss_type="discrete",  # Use discrete loss
    # ... other parameters
)
```

## ✨ Benefits

### Flow Matching Loss
- **Theoretical Foundation**: Based on state-of-the-art Flow Matching theory
- **Better Gradients**: Potentially improved gradient flow during training
- **Principled Approach**: Mathematically grounded noise-to-data transition

### Discrete Diffusion Loss
- **Minimal Changes**: Drop-in replacement for your current loss
- **Better Masking**: Improved handling of attention masks
- **Robust Averaging**: Proper loss averaging over valid tokens only

## 🔍 Technical Details

### TimeSchedulerWrapper
- Converts your `time_scheduler` values to Flow Matching format internally
- No changes needed to your existing time scheduling logic
- Handles the conversion from timestep indices to scheduler outputs

### Loss Function Signatures
```python
# Flow Matching
loss = flow_loss_fn(logits, x_1, x_t, timesteps, attention_mask)
#                                    ^^^^^^^^^ 
#                                    timestep indices, not time values!

# Discrete Diffusion  
loss = discrete_loss_fn(logits, target, attention_mask)
#                              ^^^^^^
#                              prepared target with -100 for ignored positions
```

## 🚦 Getting Started

1. **Copy `loss_final.py`** to your project
2. **Choose your approach**: Flow Matching (more advanced) or Discrete (simpler)
3. **Update your imports** and loss calculation
4. **Test with a small batch** to ensure everything works
5. **Run full training** and compare results

## 🤝 Migration from Your Current Code

**What stays the same:**
- Your `time_scheduler` creation
- Your model architecture  
- Your data loading and preprocessing
- Your optimizer and training loop structure
- Your evaluation and generation code

**What changes:**
- Loss function import and creation
- Loss calculation (1-2 lines)
- Loss function arguments

## 📈 Expected Results

- **Better loss stability** due to improved attention mask handling
- **Potentially faster convergence** with Flow Matching approach
- **Same or better generation quality** with principled training objective
- **Easy A/B testing** between loss types

---

**Ready to use!** The code is designed to work with your existing setup with minimal modifications. Start with the discrete loss for the easiest integration, then try Flow Matching for potentially better results.