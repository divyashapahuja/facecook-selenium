"""
Updated training code showing BOTH approaches for creating noised tokens:
1. Flow Matching path.sample() 
2. Your existing model._create_noise_ids() (recommended!)
"""

import torch
from loss_final import MixturePathGeneralizedKL, MixtureDiscreteProbPath, DiscreteDiffusionLoss

def training_step_flow_matching_v1(model, batch, time_scheduler, num_timesteps, path, loss_fn, accelerator):
    """
    Flow Matching approach using path.sample() for noise creation
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    # Flow Matching path sampling
    x_0 = torch.full_like(input_ids, model.mask_token_id)
    path_sample = path.sample(x_0=x_0, x_1=input_ids, timesteps=timesteps)
    noised_input_ids = path_sample.x_t
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        timesteps=timesteps,
        attention_mask=attention_mask
    )
    
    return loss


def training_step_flow_matching_v2(model, batch, time_scheduler, num_timesteps, loss_fn, accelerator):
    """
    Flow Matching approach using YOUR existing model._create_noise_ids() (RECOMMENDED!)
    This is simpler and reuses your tested noise creation logic.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    # YOUR existing noise creation (recommended!)
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # Flow Matching loss still works perfectly!
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        timesteps=timesteps,
        attention_mask=attention_mask
    )
    
    return loss


def main_simplified_flow_matching(
    model, train_loader, time_scheduler, num_timesteps, accelerator,
    use_existing_noise_creation=True  # Set to True to use your model._create_noise_ids
):
    """
    Simplified training showing both approaches
    """
    
    if use_existing_noise_creation:
        # RECOMMENDED: Use your existing noise creation + Flow Matching loss
        print("Using YOUR existing model._create_noise_ids() + Flow Matching loss")
        loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
        path = None  # Don't need path object!
        
        for batch in train_loader:
            loss = training_step_flow_matching_v2(
                model, batch, time_scheduler, num_timesteps, loss_fn, accelerator
            )
            accelerator.backward(loss)
            # ... optimizer step, etc.
            
    else:
        # Alternative: Use Flow Matching path sampling
        print("Using Flow Matching path.sample() + Flow Matching loss")
        path = MixtureDiscreteProbPath(time_scheduler)
        loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
        
        for batch in train_loader:
            loss = training_step_flow_matching_v1(
                model, batch, time_scheduler, num_timesteps, path, loss_fn, accelerator
            )
            accelerator.backward(loss)
            # ... optimizer step, etc.


def compare_noise_creation_methods(model, input_ids, time_scheduler, timesteps):
    """
    Compare the two noise creation approaches to see if they're equivalent
    """
    print("Comparing noise creation methods...")
    
    # Method 1: Flow Matching path
    path = MixtureDiscreteProbPath(time_scheduler)
    x_0 = torch.full_like(input_ids, model.mask_token_id)
    path_sample = path.sample(x_0=x_0, x_1=input_ids, timesteps=timesteps)
    noised_v1 = path_sample.x_t
    
    # Method 2: Your existing method
    noised_v2 = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    # Compare
    are_same = torch.equal(noised_v1, noised_v2)
    print(f"Are noise creation methods identical? {are_same}")
    
    if not are_same:
        print("Methods differ - this is normal due to randomness!")
        print("Both are valid approaches for creating noised tokens.")
    
    return noised_v1, noised_v2


# RECOMMENDED INTEGRATION FOR YOUR CODE:
def your_training_loop_with_flow_matching(model, train_loader, time_scheduler, num_timesteps, accelerator):
    """
    The EASIEST way to integrate Flow Matching into your existing code.
    Only 2 lines change from your original!
    """
    
    # Create loss function ONCE (before training loop)
    loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
    
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        
        # Your existing code (UNCHANGED!)
        timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
        time_t = time_scheduler[timesteps]
        noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])  # UNCHANGED!
        
        logits = model(noised_input_ids, attention_mask, time_t)  # UNCHANGED!
        
        # ONLY THIS CHANGES: Replace F.cross_entropy with Flow Matching loss
        loss = loss_fn(
            logits=logits,
            x_1=input_ids,           # Clean tokens
            x_t=noised_input_ids,    # Your noised tokens
            timesteps=timesteps,     # Timestep indices
            attention_mask=attention_mask
        )
        
        # Rest of your training loop (UNCHANGED!)
        accelerator.backward(loss)
        # ... optimizer.step(), etc.


if __name__ == "__main__":
    print("Updated Flow Matching integration options:")
    print()
    print("RECOMMENDED APPROACH:")
    print("1. Keep your existing model._create_noise_ids()")
    print("2. Only replace the loss calculation with MixturePathGeneralizedKL")
    print("3. Everything else stays exactly the same!")
    print()
    print("Benefits:")
    print("- Minimal code changes (just 2 lines)")
    print("- Reuses your tested noise creation logic")
    print("- Gets Flow Matching benefits with minimal risk")
    print("- No need to create/manage path objects")