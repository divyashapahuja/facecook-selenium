# Simple Flow Matching Integration Guide

This guide shows how to integrate Flow Matching loss into your existing discrete diffusion training with **minimal changes**.

## 🎯 Key Points

- **Use your existing `model._create_noise_ids()`** - no need for Flow Matching path sampling
- **Keep your existing `time_scheduler`** - it works directly with Flow Matching
- **Only 2 lines change** in your training loop
- **Two options**: Flow Matching loss or improved discrete loss

## 📋 Step-by-Step Integration

### Step 1: Import the Loss Functions

```python
from loss_simple import MixturePathGeneralizedKL, DiscreteDiffusionLoss
```

### Step 2: Choose Your Approach

#### Option A: Flow Matching Loss (Recommended)

```python
# Create loss function ONCE (before training loop)
loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")

# In your training loop - ONLY THE LOSS CALCULATION CHANGES:
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Your existing code (UNCHANGED!)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])  # UNCHANGED!
    
    logits = model(noised_input_ids, attention_mask, time_t)  # UNCHANGED!
    
    # NEW: Replace F.cross_entropy with Flow Matching loss
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,           # Clean tokens
        x_t=noised_input_ids,    # Your noised tokens
        timesteps=timesteps,     # Timestep indices
        attention_mask=attention_mask
    )
    
    # Rest of your training loop (UNCHANGED!)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

#### Option B: Improved Discrete Loss (Minimal Changes)

```python
# Create loss function ONCE (before training loop)
loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")

# In your training loop - ONLY THE LOSS CALCULATION CHANGES:
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Your existing code (UNCHANGED!)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Your existing target preparation (UNCHANGED!)
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # NEW: Better loss calculation
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    # Rest unchanged
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## 🔄 Complete Example: Modifying Your Training Function

Here's how to modify your existing `main()` function:

```python
def main(
    model, train_dataset, val_dataset,
    # ... your existing parameters ...
    loss_type="flow_matching",  # NEW: "flow_matching" or "discrete"
):
    # ... your existing setup code (accelerator, dataloaders, optimizer) ...
    
    # Your existing time scheduler (UNCHANGED!)
    time_scheduler = torch.linspace(
        1 / num_timesteps,
        1,
        steps=num_timesteps,
        dtype=torch.float32,
        device=accelerator.device,
    )
    
    # NEW: Create loss function ONCE before training loop
    if loss_type == "flow_matching":
        loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
        if accelerator.is_main_process:
            print("Using Flow Matching loss")
    else:  # discrete
        loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")
        if accelerator.is_main_process:
            print("Using improved discrete loss")
    
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            # Your existing timestep sampling (UNCHANGED!)
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]
            
            # Your existing noise creation (UNCHANGED!)
            noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
            
            logits = model(noised_input_ids, attention_mask, time_t)
            
            # NEW: Choose loss calculation based on loss_type
            if loss_type == "flow_matching":
                loss = loss_fn(
                    logits=logits,
                    x_1=input_ids,
                    x_t=noised_input_ids,
                    timesteps=timesteps,
                    attention_mask=attention_mask
                )
            else:  # discrete
                # Your existing target preparation
                target = input_ids.clone()
                target[noised_input_ids != model.mask_token_id] = -100
                target[attention_mask == 0] = -100
                
                loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
            
            # Rest of your training loop (UNCHANGED!)
            accelerator.backward(loss)
            # ... gradient clipping, optimizer step, logging, etc. ...
```

## 📊 What Changes vs What Stays the Same

### ✅ What Stays EXACTLY the Same:
- Your `time_scheduler` creation
- Your `model._create_noise_ids()` method
- Your model architecture and forward pass
- Your data loading and preprocessing  
- Your optimizer, scheduler, and training loop structure
- Your evaluation and generation code
- Your logging and checkpointing

### 🔄 What Changes (Minimal):
- **Import**: Add `from loss_simple import ...`
- **Loss creation**: Create loss function once before training loop
- **Loss calculation**: Replace `F.cross_entropy(...)` with `loss_fn(...)`

## 🚀 Benefits

### Flow Matching Loss:
- **Theoretical foundation**: Based on state-of-the-art Flow Matching theory
- **Better gradients**: Potentially improved gradient flow
- **Principled approach**: Mathematically grounded noise-to-data transition

### Improved Discrete Loss:
- **Better masking**: More robust attention mask handling
- **Proper averaging**: Loss averaged only over valid tokens
- **Drop-in replacement**: Minimal changes from your current approach

## 🔧 Testing Your Integration

1. **Start small**: Test with a small batch to ensure no errors
2. **Compare losses**: Run both old and new loss on the same batch to compare values
3. **Monitor training**: Watch for stable loss curves and reasonable values
4. **A/B test**: Try both loss types to see which works better for your model

## 💡 Pro Tips

1. **Use your existing noise creation** - `model._create_noise_ids()` is perfect!
2. **Create loss function once** - Don't create it inside the training loop
3. **Start with discrete loss** - It's the easiest to integrate and test
4. **Try Flow Matching after** - Once discrete loss is working, experiment with Flow Matching
5. **Keep your time_scheduler** - It works perfectly with both approaches

---

**That's it!** Your integration should be working with just these minimal changes. The code is designed to work seamlessly with your existing discrete diffusion training setup.