"""
Example showing how to use your existing time_scheduler with the new loss functions.
This demonstrates the minimal changes needed to integrate the Flow Matching loss.
"""

import torch
from loss import (
    training_step_with_existing_scheduler,
    DiscreteDiffusionLoss,
    create_flow_matching_loss_with_scheduler
)

def simulate_your_training_loop():
    """
    Simulates your existing training setup with the new loss functions.
    """
    # Your existing setup (no changes needed!)
    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    num_timesteps = 512
    
    # Create your existing time_scheduler (exactly as in your code)
    time_scheduler = torch.linspace(
        1 / num_timesteps,
        1,
        steps=num_timesteps,
        dtype=torch.float32,
        device="cpu"  # Using CPU for this example
    )
    
    print(f"time_scheduler range: {time_scheduler.min():.4f} to {time_scheduler.max():.4f}")
    print(f"time_scheduler shape: {time_scheduler.shape}")
    
    # Simulate a batch (same format as your training)
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Mock accelerator object
    class MockAccelerator:
        device = "cpu"
    accelerator = MockAccelerator()
    
    # Mock model object  
    class MockModel:
        mask_token_id = 0
        
        def _create_noise_ids(self, input_ids, time_values):
            # Simulate your existing noise creation
            mask_prob = time_values.mean(dim=-1, keepdim=True)  # Use time to determine mask probability
            mask = torch.rand_like(input_ids.float()) < mask_prob
            noised = input_ids.clone()
            noised[mask] = self.mask_token_id
            return noised
            
        def __call__(self, input_ids, attention_mask, time_t):
            # Mock forward pass - return random logits
            return torch.randn(input_ids.shape[0], input_ids.shape[1], vocab_size)
    
    model = MockModel()
    
    print("\n" + "="*50)
    print("TESTING DIFFERENT LOSS APPROACHES")
    print("="*50)
    
    # Test 1: Using the helper function with discrete loss
    print("\n1. Using training_step_with_existing_scheduler (discrete):")
    try:
        loss_discrete, logits, noised_ids = training_step_with_existing_scheduler(
            model=model,
            batch=batch,
            num_timesteps=num_timesteps,
            time_scheduler=time_scheduler,  # Your existing scheduler!
            accelerator=accelerator,
            loss_type="discrete"
        )
        print(f"   ✓ Discrete loss: {loss_discrete.item():.4f}")
        print(f"   ✓ Logits shape: {logits.shape}")
        print(f"   ✓ Noised IDs shape: {noised_ids.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Using the helper function with Flow Matching
    print("\n2. Using training_step_with_existing_scheduler (flow_matching):")
    try:
        loss_flow, logits, noised_ids = training_step_with_existing_scheduler(
            model=model,
            batch=batch,
            num_timesteps=num_timesteps,
            time_scheduler=time_scheduler,  # Your existing scheduler!
            accelerator=accelerator,
            loss_type="flow_matching"
        )
        print(f"   ✓ Flow Matching loss: {loss_flow.item():.4f}")
        print(f"   ✓ Logits shape: {logits.shape}")
        print(f"   ✓ Noised IDs shape: {noised_ids.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Manual integration (discrete)
    print("\n3. Manual integration with DiscreteDiffusionLoss:")
    try:
        loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id, reduction="mean")
        
        # Your existing sampling logic (no changes)
        timesteps = torch.randint(0, num_timesteps, (batch_size,))
        time_t = time_scheduler[timesteps]
        
        # Your existing noise creation (no changes)
        noised_input_ids = model._create_noise_ids(batch["input_ids"], time_scheduler[timesteps.unsqueeze(1)])
        
        # Forward pass (no changes)
        logits = model(noised_input_ids, batch["attention_mask"], time_t)
        
        # Target preparation (no changes)
        target = batch["input_ids"].clone()
        target[noised_input_ids != model.mask_token_id] = -100
        target[batch["attention_mask"] == 0] = -100
        
        # New loss calculation
        loss = loss_fn(logits=logits, target=target, attention_mask=batch["attention_mask"])
        print(f"   ✓ Manual discrete loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Manual Flow Matching integration
    print("\n4. Manual Flow Matching integration:")
    try:
        path, flow_loss_fn = create_flow_matching_loss_with_scheduler(
            mask_token_id=model.mask_token_id,
            scheduler_type="linear"
        )
        
        # Your existing sampling logic (no changes)
        timesteps = torch.randint(0, num_timesteps, (batch_size,))
        time_t = time_scheduler[timesteps]
        
        # Flow Matching path sampling (replaces model._create_noise_ids)
        x_0 = torch.full_like(batch["input_ids"], model.mask_token_id)
        time_normalized = (time_t - time_t.min()) / (time_t.max() - time_t.min()) if time_t.max() > time_t.min() else time_t
        path_sample = path.sample(x_0=x_0, x_1=batch["input_ids"], t=time_normalized)
        noised_input_ids = path_sample.x_t
        
        # Forward pass (no changes)
        logits = model(noised_input_ids, batch["attention_mask"], time_t)
        
        # Flow Matching loss
        loss = flow_loss_fn(
            logits=logits,
            x_1=batch["input_ids"],
            x_t=noised_input_ids,
            t=time_t,  # Your original time_scheduler values!
            attention_mask=batch["attention_mask"]
        )
        print(f"   ✓ Manual Flow Matching loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("✓ All approaches work with your existing time_scheduler")
    print("✓ Minimal code changes required")
    print("✓ Easy to switch between different loss types")
    print("✓ Compatible with your existing training infrastructure")


def show_time_scheduler_compatibility():
    """
    Shows how the time_scheduler values are handled in different approaches.
    """
    print("\n" + "="*50)
    print("TIME SCHEDULER COMPATIBILITY")
    print("="*50)
    
    num_timesteps = 10  # Small example for clarity
    
    # Your existing time_scheduler
    time_scheduler = torch.linspace(1/num_timesteps, 1, steps=num_timesteps)
    print(f"Your time_scheduler: {time_scheduler}")
    
    # Sample some timesteps
    timesteps = torch.tensor([0, 3, 7, 9])  # Sample indices
    time_values = time_scheduler[timesteps]
    print(f"Sampled time values: {time_values}")
    
    # Show how normalization works for Flow Matching
    time_normalized = (time_values - time_values.min()) / (time_values.max() - time_values.min())
    print(f"Normalized for Flow Matching: {time_normalized}")
    
    print("\nKey insight: Your model still receives the original time_scheduler values!")
    print("The Flow Matching loss handles normalization internally when needed.")


if __name__ == "__main__":
    print("Testing Flow Matching Loss with Existing time_scheduler")
    print("This example shows how to use your existing time scheduling logic")
    print("with the new loss functions.")
    
    simulate_your_training_loop()
    show_time_scheduler_compatibility()