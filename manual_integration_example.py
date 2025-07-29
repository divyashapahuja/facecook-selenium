"""
Manual integration example showing how to use your existing time_scheduler
with more control over the loss calculation.
"""

import torch
import torch.nn.functional as F
from loss import (
    DiscreteDiffusionLoss,
    MixturePathGeneralizedKL,
    MixtureDiscreteProbPath,
    LinearScheduler,
    create_flow_matching_loss_with_scheduler
)

def manual_training_step_discrete(model, batch, num_timesteps, time_scheduler, accelerator):
    """
    Manual training step using DiscreteDiffusionLoss with your existing time_scheduler.
    This requires minimal changes to your original code.
    """
    # Initialize the loss function (do this once outside the training loop in practice)
    loss_fn = DiscreteDiffusionLoss(
        mask_token_id=getattr(model, 'mask_token_id', 0), 
        reduction="mean"
    )
    
    # YOUR EXISTING CODE (unchanged):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]  # Your existing time sampling!
    
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # ONLY CHANGE: Replace F.cross_entropy with new loss
    # OLD: loss = F.cross_entropy(input=logits.transpose(-1, -2), target=target, reduction="mean")
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    return loss, logits, noised_input_ids


def manual_training_step_flow_matching(model, batch, num_timesteps, time_scheduler, accelerator):
    """
    Manual training step using Flow Matching loss with your existing time_scheduler.
    This requires more changes but gives you full control.
    """
    # Initialize Flow Matching components (do this once outside the training loop in practice)
    path, loss_fn = create_flow_matching_loss_with_scheduler(
        mask_token_id=getattr(model, 'mask_token_id', 0),
        scheduler_type="linear"
    )
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # YOUR EXISTING TIME SAMPLING (unchanged)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]  # Your existing time sampling!
    
    # MODIFIED: Use Flow Matching path sampling instead of model._create_noise_ids
    x_0 = torch.full_like(input_ids, getattr(model, 'mask_token_id', 0))  # All mask tokens
    
    # Normalize time for path sampling (Flow Matching needs [0,1])
    time_normalized = (time_t - time_t.min()) / (time_t.max() - time_t.min()) if time_t.max() > time_t.min() else time_t
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_normalized)
    noised_input_ids = path_sample.x_t
    
    # Forward pass with your original time values
    logits = model(noised_input_ids, attention_mask, time_t)  # Still use original time_t!
    
    # Flow Matching loss (automatically handles time normalization)
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,      # clean tokens
        x_t=noised_input_ids,  # noised tokens  
        t=time_t,           # Your original time_scheduler values!
        attention_mask=attention_mask
    )
    
    return loss, logits, noised_input_ids


def compare_approaches_example():
    """
    Example comparing your original approach with the new loss functions.
    """
    # Setup
    batch_size, seq_len, vocab_size = 4, 16, 1000
    num_timesteps = 512
    
    # Your existing time_scheduler (unchanged)
    time_scheduler = torch.linspace(1/num_timesteps, 1, steps=num_timesteps, dtype=torch.float32)
    
    # Mock data
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Mock model
    class MockModel:
        mask_token_id = 0
        
        def _create_noise_ids(self, input_ids, time_values):
            # Your existing noise creation logic
            mask_prob = time_values.mean(dim=-1, keepdim=True)
            mask = torch.rand_like(input_ids.float()) < mask_prob
            noised = input_ids.clone()
            noised[mask] = self.mask_token_id
            return noised
            
        def __call__(self, input_ids, attention_mask, time_t):
            return torch.randn(batch_size, seq_len, vocab_size)
    
    model = MockModel()
    
    # Mock accelerator
    class MockAccelerator:
        device = "cpu"
    accelerator = MockAccelerator()
    
    print("Comparing different approaches with your existing time_scheduler:")
    print("="*70)
    
    # Original approach
    print("\n1. ORIGINAL APPROACH:")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (batch_size,))
    time_t = time_scheduler[timesteps]
    print(f"   Time values: {time_t}")
    
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    logits = model(noised_input_ids, attention_mask, time_t)
    
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    original_loss = F.cross_entropy(
        input=logits.transpose(-1, -2),
        target=target,
        reduction="mean"
    )
    print(f"   Original loss: {original_loss.item():.4f}")
    
    # New discrete approach
    print("\n2. DISCRETE DIFFUSION LOSS (minimal changes):")
    loss_discrete, _, _ = manual_training_step_discrete(model, batch, num_timesteps, time_scheduler, accelerator)
    print(f"   Discrete loss: {loss_discrete.item():.4f}")
    print("   ✓ Uses your exact same time_scheduler")
    print("   ✓ Uses your exact same noise creation")
    print("   ✓ Only changes the loss calculation")
    
    # Flow Matching approach  
    print("\n3. FLOW MATCHING LOSS:")
    loss_flow, _, _ = manual_training_step_flow_matching(model, batch, num_timesteps, time_scheduler, accelerator)
    print(f"   Flow Matching loss: {loss_flow.item():.4f}")
    print("   ✓ Uses your exact same time_scheduler")
    print("   ✓ Model still receives original time values")
    print("   ✓ Automatic time normalization for Flow Matching")


def detailed_manual_integration():
    """
    Detailed example showing exactly what changes in your training loop.
    """
    print("\n" + "="*70)
    print("DETAILED MANUAL INTEGRATION")
    print("="*70)
    
    print("\nOption A: Minimal changes with DiscreteDiffusionLoss")
    print("-" * 50)
    
    print("""
# At the top of your file, add:
from loss import DiscreteDiffusionLoss

# Before your training loop, initialize:
loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")

# In your training loop, replace this block:
def your_original_training_step(model, batch, num_timesteps, time_scheduler, accelerator):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]                    # ← UNCHANGED
    
    noised_input_ids = model._create_noise_ids(          # ← UNCHANGED
        input_ids, time_scheduler[timesteps.unsqueeze(1)]
    )
    
    logits = model(noised_input_ids, attention_mask, time_t)  # ← UNCHANGED
    
    target = input_ids.clone()                           # ← UNCHANGED
    target[noised_input_ids != model.mask_token_id] = -100  # ← UNCHANGED
    target[attention_mask == 0] = -100                   # ← UNCHANGED
    
    # OLD:
    # loss = F.cross_entropy(input=logits.transpose(-1, -2), target=target, reduction="mean")
    
    # NEW (only change):
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    return loss
""")
    
    print("\nOption B: Flow Matching with time_scheduler")
    print("-" * 50)
    
    print("""
# At the top of your file, add:
from loss import create_flow_matching_loss_with_scheduler

# Before your training loop, initialize:
path, flow_loss_fn = create_flow_matching_loss_with_scheduler(
    mask_token_id=model.mask_token_id,
    scheduler_type="linear"
)

# In your training loop:
def your_flow_matching_training_step(model, batch, num_timesteps, time_scheduler, accelerator):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]                    # ← UNCHANGED (your original time!)
    
    # MODIFIED: Use Flow Matching path sampling
    x_0 = torch.full_like(input_ids, model.mask_token_id)
    time_normalized = (time_t - time_t.min()) / (time_t.max() - time_t.min())
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_normalized)
    noised_input_ids = path_sample.x_t
    
    logits = model(noised_input_ids, attention_mask, time_t)  # ← Still use original time_t!
    
    # Flow Matching loss
    loss = flow_loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        t=time_t,  # ← Your original time_scheduler values!
        attention_mask=attention_mask
    )
    
    return loss
""")


if __name__ == "__main__":
    print("Manual Integration Examples with Your Existing time_scheduler")
    print("This shows how to manually integrate the new loss functions")
    print("while keeping your existing time scheduling logic.")
    
    compare_approaches_example()
    detailed_manual_integration()
    
    print("\n" + "="*70)  
    print("SUMMARY")
    print("="*70)
    print("✓ Your time_scheduler stays exactly the same")
    print("✓ Your model receives the same time values")
    print("✓ Choose between minimal changes (discrete) or full Flow Matching")
    print("✓ Easy to A/B test between approaches")
    print("✓ All approaches work with your existing training infrastructure")