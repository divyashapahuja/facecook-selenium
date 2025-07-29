#!/usr/bin/env python3
"""
Test script for the generalized flow matching loss functions.
This script verifies that the loss functions work correctly with sample data.
"""

import torch
import torch.nn.functional as F
from loss import (
    GeneralizedKL, 
    MixturePathGeneralizedKL, 
    MixtureDiscreteProbPath,
    create_flow_matching_schedulers
)


def test_generalized_kl():
    """Test the GeneralizedKL loss function."""
    print("Testing GeneralizedKL...")
    
    # Create sample data
    batch_size, vocab_size, seq_len = 4, 1000, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create time-dependent coefficients
    alpha_t = torch.tensor(0.7, device=device)
    sigma_t = torch.tensor(0.3, device=device)  
    d_alpha_t = torch.tensor(1.0, device=device)
    d_sigma_t = torch.tensor(-1.0, device=device)
    
    # Create loss function
    loss_fn = GeneralizedKL(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        d_alpha_t=d_alpha_t,
        d_sigma_t=d_sigma_t,
        reduction="mean"
    )
    
    # Sample data
    logits = torch.randn(batch_size, vocab_size, seq_len, device=device)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Add some -100 tokens (ignored)
    target[:, -5:] = -100
    
    # Compute loss
    loss = loss_fn(logits, target)
    
    print(f"✓ GeneralizedKL loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"
    

def test_mixture_path_generalized_kl():
    """Test the MixturePathGeneralizedKL loss function."""
    print("Testing MixturePathGeneralizedKL...")
    
    # Create sample data
    batch_size, vocab_size, seq_len = 4, 1000, 32
    num_paths = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create time-dependent coefficients for multiple paths
    alpha_t = torch.tensor([0.5, 0.7, 0.9], device=device)
    sigma_t = torch.tensor([0.5, 0.3, 0.1], device=device)
    d_alpha_t = torch.ones(num_paths, device=device)
    d_sigma_t = -torch.ones(num_paths, device=device)
    
    # Create loss function
    loss_fn = MixturePathGeneralizedKL(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        d_alpha_t=d_alpha_t,
        d_sigma_t=d_sigma_t,
        reduction="mean"
    )
    
    # Sample data
    logits = torch.randn(batch_size, vocab_size, seq_len, device=device)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Add some -100 tokens (ignored)
    target[:, -5:] = -100
    
    # Compute loss
    loss = loss_fn(logits, target)
    
    print(f"✓ MixturePathGeneralizedKL loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_mixture_discrete_prob_path():
    """Test the MixtureDiscreteProbPath loss function."""
    print("Testing MixtureDiscreteProbPath...")
    
    # Create sample data
    batch_size, vocab_size, seq_len = 4, 1000, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_token_id = vocab_size - 1  # Use last token as mask
    
    # Simple time scheduler
    def time_scheduler(timesteps):
        return timesteps.float() / 100.0  # Normalize to [0, 1]
    
    # Create loss function
    loss_fn = MixtureDiscreteProbPath(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        time_scheduler=time_scheduler,
        reduction="mean"
    )
    
    # Sample data
    logits = torch.randn(batch_size, vocab_size, seq_len, device=device)
    target = torch.randint(0, vocab_size-1, (batch_size, seq_len), device=device)  # Avoid mask token
    timesteps = torch.randint(0, 100, (batch_size,), device=device)
    
    # Add some -100 tokens (ignored)
    target[:, -5:] = -100
    
    # Compute loss
    loss = loss_fn(logits, target, timesteps)
    
    print(f"✓ MixtureDiscreteProbPath loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_schedulers():
    """Test different scheduler types."""
    print("Testing schedulers...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_timesteps = 100
    
    for scheduler_type in ["linear", "cosine", "exponential"]:
        print(f"  Testing {scheduler_type} scheduler...")
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = create_flow_matching_schedulers(
            num_timesteps, device, scheduler_type
        )
        
        # Check shapes
        assert alpha_t.shape == (num_timesteps,), f"Wrong alpha_t shape: {alpha_t.shape}"
        assert sigma_t.shape == (num_timesteps,), f"Wrong sigma_t shape: {sigma_t.shape}"
        assert d_alpha_t.shape == (num_timesteps,), f"Wrong d_alpha_t shape: {d_alpha_t.shape}"
        assert d_sigma_t.shape == (num_timesteps,), f"Wrong d_sigma_t shape: {d_sigma_t.shape}"
        
        # Check values are finite
        assert torch.all(torch.isfinite(alpha_t)), "alpha_t contains non-finite values"
        assert torch.all(torch.isfinite(sigma_t)), "sigma_t contains non-finite values"
        assert torch.all(torch.isfinite(d_alpha_t)), "d_alpha_t contains non-finite values"
        assert torch.all(torch.isfinite(d_sigma_t)), "d_sigma_t contains non-finite values"
        
        # Check reasonable ranges
        assert torch.all(alpha_t >= 0), "alpha_t should be non-negative"
        assert torch.all(sigma_t >= 0), "sigma_t should be non-negative"
        
        print(f"    ✓ {scheduler_type} scheduler passed all checks")


def test_integration():
    """Test integration with realistic model-like scenario."""
    print("Testing integration scenario...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, vocab_size, seq_len = 8, 5000, 64
    num_timesteps = 256
    
    # Create schedulers
    alpha_t, sigma_t, d_alpha_t, d_sigma_t = create_flow_matching_schedulers(
        num_timesteps, device, "linear"
    )
    
    # Create loss function
    loss_fn = MixturePathGeneralizedKL(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        d_alpha_t=d_alpha_t,
        d_sigma_t=d_sigma_t,
        reduction="mean"
    )
    
    # Simulate training step
    logits = torch.randn(batch_size, vocab_size, seq_len, device=device)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Add padding tokens
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    attention_mask[:, -10:] = False  # Last 10 tokens are padding
    target[~attention_mask] = -100
    
    # Compute loss
    loss = loss_fn(logits, target)
    
    print(f"✓ Integration test loss: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print("✓ Backward pass successful")
    
    # Check gradients exist
    assert logits.grad is not None, "Gradients should exist after backward pass"
    assert torch.all(torch.isfinite(logits.grad)), "Gradients should be finite"


def test_comparison_with_cross_entropy():
    """Compare with standard cross entropy to ensure reasonable behavior."""
    print("Comparing with cross entropy...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, vocab_size, seq_len = 4, 1000, 32
    
    # Create sample data
    logits = torch.randn(batch_size, vocab_size, seq_len, device=device, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target[:, -5:] = -100  # Add some ignored tokens
    
    # Standard cross entropy
    ce_loss = F.cross_entropy(
        logits.transpose(-1, -2).reshape(-1, vocab_size),
        target.reshape(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    # Flow matching loss (should reduce to similar values when coefficients are appropriate)
    alpha_t = torch.tensor(1.0, device=device)  # Full data weight
    sigma_t = torch.tensor(0.0, device=device)  # No noise
    d_alpha_t = torch.tensor(0.0, device=device)  # No derivative correction
    d_sigma_t = torch.tensor(0.0, device=device)
    
    flow_loss_fn = GeneralizedKL(
        alpha_t=alpha_t,
        sigma_t=sigma_t,
        d_alpha_t=d_alpha_t,
        d_sigma_t=d_sigma_t,
        reduction="mean"
    )
    
    flow_loss = flow_loss_fn(logits, target)
    
    print(f"Cross entropy loss: {ce_loss.item():.4f}")
    print(f"Flow matching loss: {flow_loss.item():.4f}")
    
    # They should be reasonably close when flow coefficients are set to reduce to CE
    ratio = flow_loss.item() / ce_loss.item()
    print(f"Ratio (flow/ce): {ratio:.4f}")
    
    # Should be within reasonable range (the flow loss includes additional terms)
    assert 0.5 < ratio < 2.0, f"Loss ratio {ratio} seems unreasonable"


def main():
    """Run all tests."""
    print("Running loss function tests...\n")
    
    try:
        test_generalized_kl()
        print()
        
        test_mixture_path_generalized_kl()
        print()
        
        test_mixture_discrete_prob_path()
        print()
        
        test_schedulers()
        print()
        
        test_integration()
        print()
        
        test_comparison_with_cross_entropy()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()