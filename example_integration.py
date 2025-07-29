"""
Example integration showing exactly how to modify your existing training code.
This demonstrates the minimal changes needed to add Flow Matching loss.
"""

import torch
from loss_simple import MixturePathGeneralizedKL, DiscreteDiffusionLoss

def your_original_training_step(model, batch, time_scheduler, num_timesteps, accelerator):
    """
    Your ORIGINAL training step (for reference)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    target = input_ids.clone()
    target[noised_input_ids != model.mask_token_id] = -100
    target[attention_mask == 0] = -100
    
    # Your original loss
    loss = torch.nn.functional.cross_entropy(
        input=logits.transpose(-1, -2),
        target=target,
        reduction="mean",
    )
    
    return loss


def flow_matching_training_step(model, batch, time_scheduler, num_timesteps, loss_fn, accelerator):
    """
    NEW: Training step with Flow Matching loss
    Only the loss calculation changes!
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    
    # Your existing code (UNCHANGED!)
    timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
    time_t = time_scheduler[timesteps]
    noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
    
    logits = model(noised_input_ids, attention_mask, time_t)
    
    # NEW: Flow Matching loss (replaces F.cross_entropy)
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,           # Clean tokens
        x_t=noised_input_ids,    # Your noised tokens
        timesteps=timesteps,     # Timestep indices
        attention_mask=attention_mask
    )
    
    return loss


def discrete_training_step(model, batch, time_scheduler, num_timesteps, loss_fn, accelerator):
    """
    NEW: Training step with improved discrete loss
    Very similar to your original, just better loss calculation
    """
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
    
    return loss


def modified_main_function(
    model, train_dataset, val_dataset,
    batch_size=32,
    num_timesteps=512,
    loss_type="flow_matching",  # NEW parameter
    # ... your other existing parameters ...
):
    """
    Your main training function with minimal modifications
    """
    
    # ... your existing setup code (accelerator, dataloaders, optimizer) ...
    # This is just a skeleton - use your actual setup code
    
    accelerator = None  # Your accelerator setup
    train_loader = None  # Your train_loader setup
    optimizer = None    # Your optimizer setup
    
    # Your existing time scheduler (UNCHANGED!)
    time_scheduler = torch.linspace(
        1 / num_timesteps,
        1,
        steps=num_timesteps,
        dtype=torch.float32,
        device=accelerator.device if accelerator else torch.device('cpu'),
    )
    
    # NEW: Create loss function ONCE before training loop
    if loss_type == "flow_matching":
        loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
        print("Using Flow Matching loss")
    elif loss_type == "discrete":
        loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")
        print("Using improved discrete loss")
    else:
        loss_fn = None
        print("Using original F.cross_entropy loss")
    
    # Your existing training loop with minimal changes
    step = 0
    for epoch in range(10):  # Your actual epoch count
        model.train()
        for batch in train_loader:
            
            # Choose training step based on loss type
            if loss_type == "flow_matching":
                loss = flow_matching_training_step(
                    model, batch, time_scheduler, num_timesteps, loss_fn, accelerator
                )
            elif loss_type == "discrete":
                loss = discrete_training_step(
                    model, batch, time_scheduler, num_timesteps, loss_fn, accelerator
                )
            else:
                loss = your_original_training_step(
                    model, batch, time_scheduler, num_timesteps, accelerator
                )
            
            # Rest of your training loop (UNCHANGED!)
            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
                
            optimizer.step()
            optimizer.zero_grad()
            
            # Your logging, evaluation, etc. (UNCHANGED!)
            step += 1


def quick_integration_example():
    """
    The QUICKEST way to integrate - just modify your loss calculation
    """
    
    # Assume you have these from your existing code:
    # model, time_scheduler, num_timesteps, accelerator
    
    # 1. Create loss function ONCE
    loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
    
    # 2. In your training loop, replace this:
    """
    # OLD:
    loss = F.cross_entropy(
        input=logits.transpose(-1, -2),
        target=target,
        reduction="mean",
    )
    """
    
    # With this:
    """
    # NEW:
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        timesteps=timesteps,
        attention_mask=attention_mask
    )
    """
    
    print("That's it! Only 2 lines change in your training loop.")


if __name__ == "__main__":
    print("Flow Matching Integration Examples")
    print("=" * 40)
    print()
    print("Key changes to your existing training:")
    print("1. Import: from loss_simple import MixturePathGeneralizedKL, DiscreteDiffusionLoss")
    print("2. Create loss function once before training loop")
    print("3. Replace F.cross_entropy with new loss function")
    print("4. Everything else stays exactly the same!")
    print()
    print("Two options:")
    print("- Flow Matching: MixturePathGeneralizedKL (recommended)")
    print("- Discrete: DiscreteDiffusionLoss (minimal changes)")
    print()
    print("Both work with your existing:")
    print("- time_scheduler")
    print("- model._create_noise_ids()")
    print("- Training loop structure")
    print("- Model architecture")
    print("- Everything else!")