"""
Example of how to integrate the adjusted Flow Matching loss with your discrete diffusion training.

This shows two approaches:
1. Using the Flow Matching MixturePathGeneralizedKL loss
2. Using the simplified DiscreteDiffusionLoss (closer to your original setup)
"""

import torch
import torch.nn.functional as F
from loss import (
    MixturePathGeneralizedKL, 
    MixtureDiscreteProbPath, 
    LinearScheduler,
    PolynomialConvexScheduler,
    DiscreteDiffusionLoss,
    create_time_scheduler
)

def training_step_with_flow_matching_loss(
    model, 
    batch, 
    num_timesteps, 
    accelerator,
    loss_fn=None
):
    """
    Training step using the Flow Matching generalized KL loss.
    
    This is an adapted version of your training loop that uses the Flow Matching loss.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Create scheduler and path (you can choose between LinearScheduler or PolynomialConvexScheduler)
    scheduler = LinearScheduler()  # or PolynomialConvexScheduler(n=1.0)
    path = MixtureDiscreteProbPath(scheduler)
    
    if loss_fn is None:
        loss_fn = MixturePathGeneralizedKL(path, reduction="mean")
    
    # Sample timesteps
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = timesteps.float() / num_timesteps  # Normalize to [0, 1]
    
    # Create noise (mask tokens) - this is your x_0
    mask_token_id = model.mask_token_id if hasattr(model, 'mask_token_id') else 0
    x_0 = torch.full_like(input_ids, mask_token_id)
    
    # Sample from the path: x_t ~ p_t(·|x_1) where x_1 is the clean data
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_t)
    noised_input_ids = path_sample.x_t
    
    # Forward pass
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Compute Flow Matching loss
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,  # target (clean tokens)
        x_t=noised_input_ids,  # noised tokens
        t=time_t,
        attention_mask=attention_mask
    )
    
    return loss, logits, noised_input_ids


def training_step_with_discrete_diffusion_loss(
    model, 
    batch, 
    num_timesteps, 
    time_scheduler,
    accelerator,
    loss_fn=None
):
    """
    Training step using the simplified discrete diffusion loss.
    
    This is closer to your original training setup but uses the new loss module.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    if loss_fn is None:
        mask_token_id = model.mask_token_id if hasattr(model, 'mask_token_id') else 0
        loss_fn = DiscreteDiffusionLoss(mask_token_id=mask_token_id, reduction="mean")
    
    # Sample timesteps (same as your original code)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    # Create noised inputs (same as your original code)
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    # Forward pass
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Prepare target (same as your original code)
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # Compute loss using the new loss module
    loss = loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    return loss, logits, noised_input_ids


def modified_main_function_example():
    """
    Example of how to modify your main training function to use the new loss.
    """
    # ... (your existing setup code) ...
    
    # Option 1: Use Flow Matching loss
    scheduler = LinearScheduler()  # or PolynomialConvexScheduler(n=1.0)
    path = MixtureDiscreteProbPath(scheduler)
    flow_matching_loss = MixturePathGeneralizedKL(path, reduction="mean")
    
    # Option 2: Use simplified discrete diffusion loss
    mask_token_id = 0  # Replace with your actual mask token ID
    discrete_loss = DiscreteDiffusionLoss(mask_token_id=mask_token_id, reduction="mean")
    
    # Create time scheduler (you can use the helper function)
    time_scheduler = create_time_scheduler(
        num_timesteps=512, 
        scheduler_type="linear",  # or "polynomial"
        device=accelerator.device
    )
    
    # In your training loop, replace the loss calculation with:
    # For Flow Matching approach:
    # loss, logits, noised_input_ids = training_step_with_flow_matching_loss(
    #     model, batch, num_timesteps, accelerator, flow_matching_loss
    # )
    
    # For simplified approach (closer to your original):
    # loss, logits, noised_input_ids = training_step_with_discrete_diffusion_loss(
    #     model, batch, num_timesteps, time_scheduler, accelerator, discrete_loss
    # )


def compare_losses_example():
    """
    Example comparing the original cross-entropy loss with the new losses.
    """
    # Dummy data
    batch_size, seq_len, vocab_size = 2, 10, 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Original cross-entropy approach
    target = input_ids.clone()
    mask_positions = torch.rand(batch_size, seq_len) < 0.3  # 30% masking
    target[mask_positions] = -100
    target[attention_mask == 0] = -100
    
    original_loss = F.cross_entropy(
        input=logits.transpose(-1, -2),
        target=target,
        reduction="mean"
    )
    
    # New discrete diffusion loss
    mask_token_id = 0
    discrete_loss_fn = DiscreteDiffusionLoss(mask_token_id=mask_token_id, reduction="mean")
    new_loss = discrete_loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    
    print(f"Original loss: {original_loss.item():.4f}")
    print(f"New discrete loss: {new_loss.item():.4f}")
    
    # Flow Matching loss (requires more setup)
    scheduler = LinearScheduler()
    path = MixtureDiscreteProbPath(scheduler)
    flow_loss_fn = MixturePathGeneralizedKL(path, reduction="mean")
    
    # Create noised version for Flow Matching
    time_t = torch.rand(batch_size)  # Random times in [0, 1]
    x_0 = torch.full_like(input_ids, mask_token_id)  # All mask tokens
    path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_t)
    
    flow_loss = flow_loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=path_sample.x_t,
        t=time_t,
        attention_mask=attention_mask
    )
    
    print(f"Flow Matching loss: {flow_loss.item():.4f}")


if __name__ == "__main__":
    compare_losses_example()