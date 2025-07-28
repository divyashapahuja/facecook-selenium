#!/usr/bin/env python3
"""
Example usage of the adjusted MixturePathGeneralizedKL loss module
for discrete flow matching in language models.
"""

import torch
import torch.nn.functional as F
from loss import MixturePathGeneralizedKL, MixtureDiscreteProbPath, SimplifiedFlowLoss


def create_dummy_data(batch_size=4, seq_len=10, vocab_size=1000):
    """Create dummy data for testing the loss functions"""
    
    # Create random logits (model output)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create target tokens (original clean text)
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create noised input (with some tokens replaced by mask token)
    noised_input = target.clone()
    mask_token_id = vocab_size - 1  # Use last token as mask
    
    # Randomly mask some positions
    mask_prob = 0.3
    mask_positions = torch.rand(batch_size, seq_len) < mask_prob
    noised_input[mask_positions] = mask_token_id
    
    # Create time values
    t = torch.rand(batch_size)
    
    # Create attention mask (all positions are valid in this example)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Set some positions to be ignored in loss calculation (like padding)
    target_for_loss = target.clone()
    # Ignore positions that weren't masked (only train on masked positions)
    target_for_loss[~mask_positions] = -100
    # Also ignore some positions as if they were padding
    target_for_loss[:, -2:] = -100  # Ignore last 2 positions
    
    return logits, target_for_loss, noised_input, t, attention_mask


def test_mixture_path_generalized_kl():
    """Test the MixturePathGeneralizedKL loss function"""
    print("Testing MixturePathGeneralizedKL loss...")
    
    # Create dummy data
    logits, target, noised_input, t, attention_mask = create_dummy_data()
    
    print(f"Input shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  target: {target.shape}")
    print(f"  noised_input: {noised_input.shape}")
    print(f"  t: {t.shape}")
    
    # Create the loss function
    path = MixtureDiscreteProbPath()
    loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")
    
    # Calculate loss
    loss = loss_fn(logits, target, noised_input, t)
    
    print(f"MixturePathGeneralizedKL loss: {loss.item():.4f}")
    
    # Test different reduction modes
    loss_fn_none = MixturePathGeneralizedKL(path=path, reduction="none")
    loss_none = loss_fn_none(logits, target, noised_input, t)
    print(f"Loss with reduction='none' shape: {loss_none.shape}")
    
    loss_fn_sum = MixturePathGeneralizedKL(path=path, reduction="sum")
    loss_sum = loss_fn_sum(logits, target, noised_input, t)
    print(f"Loss with reduction='sum': {loss_sum.item():.4f}")
    
    return loss


def test_simplified_flow_loss():
    """Test the SimplifiedFlowLoss function"""
    print("\nTesting SimplifiedFlowLoss...")
    
    # Create dummy data
    logits, target, noised_input, t, attention_mask = create_dummy_data()
    
    # Create the loss function
    loss_fn = SimplifiedFlowLoss(reduction="mean")
    
    # Calculate loss
    loss = loss_fn(logits, target, noised_input, t)
    
    print(f"SimplifiedFlowLoss: {loss.item():.4f}")
    
    return loss


def test_custom_scheduler():
    """Test with a custom scheduler function"""
    print("\nTesting with custom scheduler...")
    
    def custom_scheduler(t):
        from loss import SchedulerOutput
        # Custom quadratic scheduler
        alpha_t = (1 - t) ** 2
        d_alpha_t = -2 * (1 - t)
        return SchedulerOutput(alpha_t, d_alpha_t)
    
    # Create dummy data
    logits, target, noised_input, t, attention_mask = create_dummy_data()
    
    # Create path with custom scheduler
    path = MixtureDiscreteProbPath(scheduler_fn=custom_scheduler)
    loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")
    
    # Calculate loss
    loss = loss_fn(logits, target, noised_input, t)
    
    print(f"Loss with custom scheduler: {loss.item():.4f}")
    
    return loss


def compare_with_cross_entropy():
    """Compare flow losses with standard cross-entropy loss"""
    print("\nComparing with standard cross-entropy loss...")
    
    # Create dummy data
    logits, target, noised_input, t, attention_mask = create_dummy_data()
    
    # Standard cross-entropy loss (ignoring -100 positions)
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        target.view(-1), 
        ignore_index=-100,
        reduction="mean"
    )
    
    # Flow losses
    path = MixtureDiscreteProbPath()
    flow_loss = MixturePathGeneralizedKL(path=path, reduction="mean")(
        logits, target, noised_input, t
    )
    
    simple_flow_loss = SimplifiedFlowLoss(reduction="mean")(
        logits, target, noised_input, t
    )
    
    print(f"Standard Cross-Entropy Loss: {ce_loss.item():.4f}")
    print(f"MixturePathGeneralizedKL Loss: {flow_loss.item():.4f}")
    print(f"SimplifiedFlowLoss: {simple_flow_loss.item():.4f}")


def test_gradient_flow():
    """Test that gradients flow properly through the loss functions"""
    print("\nTesting gradient flow...")
    
    # Create dummy data with requires_grad=True
    logits, target, noised_input, t, attention_mask = create_dummy_data()
    logits.requires_grad_(True)
    
    # Test MixturePathGeneralizedKL
    path = MixtureDiscreteProbPath()
    loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")
    loss = loss_fn(logits, target, noised_input, t)
    
    # Backward pass
    loss.backward()
    
    print(f"Gradient norm for logits: {logits.grad.norm().item():.4f}")
    print("✓ Gradients flow properly through MixturePathGeneralizedKL")
    
    # Reset gradients and test SimplifiedFlowLoss
    logits.grad.zero_()
    loss_fn_simple = SimplifiedFlowLoss(reduction="mean")
    loss_simple = loss_fn_simple(logits, target, noised_input, t)
    loss_simple.backward()
    
    print(f"Gradient norm for logits (SimplifiedFlowLoss): {logits.grad.norm().item():.4f}")
    print("✓ Gradients flow properly through SimplifiedFlowLoss")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Adjusted Loss Module for Discrete Flow Matching")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_mixture_path_generalized_kl()
    test_simplified_flow_loss()
    test_custom_scheduler()
    compare_with_cross_entropy()
    test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)