import torch
import torch.nn.functional as F
from generalized_loss import MixturePathGeneralizedKL

# Initialize the generalized loss function
generalized_loss_fn = MixturePathGeneralizedKL(reduction="mean")

def compute_scheduler_params(time_t, num_timesteps):
    """
    Compute alpha_t and d_alpha_t based on your time scheduler.
    You'll need to adapt this based on your specific scheduler implementation.
    
    This is a placeholder - you'll need to implement this based on your 
    time_scheduler and how it defines alpha_t and its derivative.
    """
    # Example implementation - replace with your actual scheduler logic
    # This assumes a linear schedule from 1 to 0
    alpha_t = 1.0 - time_t  # This should match your scheduler's alpha_t
    d_alpha_t = -torch.ones_like(time_t)  # Derivative of alpha_t w.r.t. time
    
    return alpha_t, d_alpha_t

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
            
            # Prepare target for generalized loss (don't mask with -100)
            target = input_ids.clone()
            
            # Compute scheduler parameters
            alpha_t, d_alpha_t = compute_scheduler_params(time_t, num_timesteps)
            
            # Apply attention mask by setting loss to 0 for padding tokens
            # Create a mask for valid tokens
            valid_mask = attention_mask.unsqueeze(-1).expand_as(target)
            
            # Use generalized loss instead of cross-entropy
            loss = generalized_loss_fn(
                logits=logits,
                x_1=target,
                x_t=noised_input_ids,
                t=time_t,
                alpha_t=alpha_t,
                d_alpha_t=d_alpha_t
            )
            
            # Apply attention mask to loss
            if valid_mask is not None:
                loss = loss * valid_mask.float()
                loss = loss.sum() / valid_mask.sum()
            
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    return total_loss / max(total_batches, 1)

def training_loop_example(model, train_loader, epochs, num_timesteps, time_scheduler, accelerator):
    """Example training loop using generalized loss"""
    
    step = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            # Move batch to device efficiently (accelerator handles this automatically)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].bool()
            
            timesteps = torch.randint(0, num_timesteps, (input_ids.shape[0],), device=accelerator.device)
            time_t = time_scheduler[timesteps]
            
            noised_input_ids = model._create_noise_ids(input_ids, time_scheduler[timesteps.unsqueeze(1)])
            
            logits = model(noised_input_ids, attention_mask, time_t)
            
            # Prepare target for generalized loss
            target = input_ids.clone()
            
            if accelerator.is_main_process and step % 100 == 0:  # Reduce print frequency
                print(f"logits: {logits.shape}")
                print(f"target: {target.shape}")
            
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
            
            # Apply attention mask to loss
            valid_mask = attention_mask.unsqueeze(-1).expand_as(target)
            loss = loss * valid_mask.float()
            loss = loss.sum() / valid_mask.sum()
            
            # Use accelerator's backward pass
            accelerator.backward(loss)
            
            step += 1

# Alternative: If you want to keep the cross-entropy masking approach
def evaluate_model_with_masking(model, val_loader, num_timesteps, time_scheduler, accelerator, max_eval_batches=50):
    """Evaluate model using generalized loss with -100 masking approach"""
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
            
            target = input_ids.clone()
            
            # Create masks for valid positions
            masked_positions = noised_input_ids != model.mask_token_id
            padding_positions = attention_mask == 0
            
            # Only compute loss on tokens that were masked and are not padding
            valid_positions = ~masked_positions & ~padding_positions
            
            if valid_positions.sum() > 0:
                # Extract only valid positions
                valid_logits = logits[valid_positions]
                valid_targets = target[valid_positions]
                valid_noised = noised_input_ids[valid_positions]
                
                # Compute scheduler parameters
                alpha_t, d_alpha_t = compute_scheduler_params(time_t, num_timesteps)
                
                # Expand alpha_t and d_alpha_t for valid positions
                batch_indices = torch.arange(len(time_t), device=time_t.device)
                batch_indices = batch_indices.unsqueeze(1).expand_as(valid_positions)[valid_positions]
                valid_alpha_t = alpha_t[batch_indices]
                valid_d_alpha_t = d_alpha_t[batch_indices]
                
                # Use generalized loss
                loss = generalized_loss_fn(
                    logits=valid_logits.unsqueeze(1),  # Add sequence dimension
                    x_1=valid_targets.unsqueeze(1),
                    x_t=valid_noised.unsqueeze(1),
                    t=time_t[batch_indices],
                    alpha_t=valid_alpha_t,
                    d_alpha_t=valid_d_alpha_t
                )
            else:
                loss = torch.tensor(0.0, device=accelerator.device)
            
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    return total_loss / max(total_batches, 1)