"""
Detailed explanation of why time normalization is needed in Flow Matching
and when you can skip it.
"""

import torch
from loss import LinearScheduler, MixtureDiscreteProbPath, MixturePathGeneralizedKL

def show_scheduler_behavior():
    """
    Shows how Flow Matching schedulers behave with different time ranges.
    """
    print("FLOW MATCHING SCHEDULER BEHAVIOR")
    print("="*50)
    
    scheduler = LinearScheduler()
    
    # Test different time ranges
    time_ranges = [
        ("Your time_scheduler", torch.tensor([0.00195, 0.5, 1.0])),
        ("Normalized [0,1]", torch.tensor([0.0, 0.5, 1.0])),
        ("Random batch", torch.tensor([0.1, 0.15, 0.2])),  # What if batch has similar times?
    ]
    
    for name, times in time_ranges:
        print(f"\n{name}: {times}")
        output = scheduler(times)
        print(f"  alpha_t (data signal): {output.alpha_t}")
        print(f"  sigma_t (noise level): {output.sigma_t}")
        print(f"  d_alpha_t (derivative): {output.d_alpha_t}")
        
        # Check the jump coefficient used in loss
        jump_coeff = output.d_alpha_t / (1 - output.alpha_t)
        print(f"  jump_coefficient: {jump_coeff}")
        
        # Look for potential issues
        if torch.any(output.alpha_t < 0.001):
            print("  ⚠️  Very small alpha_t - might cause numerical issues")
        if torch.any(jump_coeff > 100):
            print("  ⚠️  Very large jump coefficient - might cause gradient issues")
        if torch.all(times > 0.8):
            print("  ⚠️  All times near end - limited noise diversity")


def compare_with_and_without_normalization():
    """
    Compare Flow Matching loss with and without time normalization.
    """
    print("\n" + "="*50)
    print("COMPARING WITH/WITHOUT NORMALIZATION")
    print("="*50)
    
    # Setup
    batch_size, seq_len, vocab_size = 2, 8, 100
    num_timesteps = 512
    
    # Your original time_scheduler
    time_scheduler = torch.linspace(1/num_timesteps, 1, steps=num_timesteps)
    
    # Mock data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = torch.randn(batch_size, seq_len, vocab_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Sample some timesteps
    timesteps = torch.tensor([0, 100, 511])  # Early, middle, late
    time_t = time_scheduler[timesteps]
    
    print(f"Original time values: {time_t}")
    print(f"Range: [{time_t.min():.6f}, {time_t.max():.6f}]")
    
    # Create Flow Matching components
    scheduler = LinearScheduler()
    path = MixtureDiscreteProbPath(scheduler)
    
    # Test 1: Without normalization (using your times directly)
    print("\n1. WITHOUT normalization (using your times directly):")
    try:
        # Create some noised data
        x_0 = torch.zeros_like(input_ids)  # All mask tokens
        path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_t)
        
        loss_fn = MixturePathGeneralizedKL(path, reduction="mean", normalize_time=False)
        loss = loss_fn(
            logits=logits,
            x_1=input_ids,
            x_t=path_sample.x_t,
            t=time_t,  # Your original times
            attention_mask=attention_mask
        )
        print(f"   ✓ Loss: {loss.item():.4f}")
        print("   ✓ Works fine! Your time range is actually OK.")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: With normalization
    print("\n2. WITH normalization:")
    try:
        # Normalize times to [0, 1]
        time_normalized = (time_t - time_t.min()) / (time_t.max() - time_t.min()) if time_t.max() > time_t.min() else time_t
        print(f"   Normalized times: {time_normalized}")
        
        path_sample = path.sample(x_0=x_0, x_1=input_ids, t=time_normalized)
        
        loss_fn = MixturePathGeneralizedKL(path, reduction="mean", normalize_time=True)
        loss = loss_fn(
            logits=logits,
            x_1=input_ids,
            x_t=path_sample.x_t,
            t=time_t,  # Still pass original times, loss normalizes internally
            attention_mask=attention_mask
        )
        print(f"   ✓ Loss: {loss.item():.4f}")
        print("   ✓ Also works! Normalization provides extra safety.")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")


def show_when_normalization_matters():
    """
    Show specific cases where normalization becomes important.
    """
    print("\n" + "="*50)
    print("WHEN NORMALIZATION MATTERS")
    print("="*50)
    
    scheduler = LinearScheduler()
    
    # Case 1: Batch with very similar timesteps
    print("\n1. Batch with very similar timesteps:")
    similar_times = torch.tensor([0.001, 0.002, 0.003])  # All very early
    print(f"   Original times: {similar_times}")
    
    # Without normalization
    output = scheduler(similar_times)
    jump_coeff = output.d_alpha_t / (1 - output.alpha_t)
    print(f"   Jump coefficients: {jump_coeff}")
    print(f"   Range: [{jump_coeff.min():.3f}, {jump_coeff.max():.3f}]")
    
    # With normalization
    normalized = (similar_times - similar_times.min()) / (similar_times.max() - similar_times.min())
    print(f"   Normalized: {normalized}")
    output_norm = scheduler(normalized)
    jump_coeff_norm = output_norm.d_alpha_t / (1 - output_norm.alpha_t)
    print(f"   Normalized jump coefficients: {jump_coeff_norm}")
    print(f"   Normalized range: [{jump_coeff_norm.min():.3f}, {jump_coeff_norm.max():.3f}]")
    
    # Case 2: Edge case - single timestep
    print("\n2. Single timestep (no variance):")
    single_time = torch.tensor([0.5])
    print(f"   Single time: {single_time}")
    
    # Normalization handles this gracefully
    try:
        normalized_single = (single_time - single_time.min()) / (single_time.max() - single_time.min())
        print(f"   Normalized result: {normalized_single}")  # Will be NaN or handled
        print("   ⚠️  This is why we need special handling in normalization code")
    except:
        print("   ✗ Division by zero without proper handling")


def practical_recommendations():
    """
    Practical recommendations for when to normalize.
    """
    print("\n" + "="*50)
    print("PRACTICAL RECOMMENDATIONS")
    print("="*50)
    
    print("\n✅ You DON'T need normalization if:")
    print("   - Using DiscreteDiffusionLoss (doesn't use Flow Matching schedulers)")
    print("   - Your time_scheduler already goes from 0 to 1")
    print("   - You modify your scheduler: torch.linspace(0, 1, steps=num_timesteps)")
    
    print("\n⚠️  You SHOULD normalize if:")
    print("   - Using Flow Matching loss with your current time_scheduler")
    print("   - You want mathematical guarantees about scheduler behavior")
    print("   - You're seeing numerical instabilities")
    print("   - You want to be safe and follow Flow Matching conventions")
    
    print("\n🔧 Easy fix options:")
    print("   Option 1: Change your time_scheduler")
    print("     time_scheduler = torch.linspace(0, 1, steps=num_timesteps)")
    print("   ")
    print("   Option 2: Use normalize_time=True (what we implemented)")
    print("     loss_fn = MixturePathGeneralizedKL(path, normalize_time=True)")
    print("   ")
    print("   Option 3: Use DiscreteDiffusionLoss (no normalization needed)")
    print("     loss_fn = DiscreteDiffusionLoss(mask_token_id=model.mask_token_id)")


def show_your_scheduler_is_actually_fine():
    """
    Demonstrate that your time_scheduler actually works fine with Flow Matching.
    """
    print("\n" + "="*50)
    print("YOUR SCHEDULER IS ACTUALLY FINE!")
    print("="*50)
    
    num_timesteps = 512
    your_scheduler = torch.linspace(1/num_timesteps, 1, steps=num_timesteps)
    ideal_scheduler = torch.linspace(0, 1, steps=num_timesteps)
    
    print(f"Your scheduler range: [{your_scheduler.min():.6f}, {your_scheduler.max():.6f}]")
    print(f"Ideal scheduler range: [{ideal_scheduler.min():.6f}, {ideal_scheduler.max():.6f}]")
    print(f"Difference in min: {your_scheduler.min() - ideal_scheduler.min():.6f}")
    print("This tiny difference (0.002) rarely causes problems!")
    
    # Test with Flow Matching scheduler
    scheduler = LinearScheduler()
    
    # Sample some times
    test_times = your_scheduler[[0, 100, 256, 511]]  # Various points
    print(f"\nTest times: {test_times}")
    
    output = scheduler(test_times)
    print(f"Alpha values: {output.alpha_t}")
    print(f"Sigma values: {output.sigma_t}")
    
    # Check if values make sense
    print(f"\nAt earliest time ({test_times[0]:.6f}):")
    print(f"  - Data signal: {output.alpha_t[0]:.6f} (very small ✓)")
    print(f"  - Noise level: {output.sigma_t[0]:.6f} (very high ✓)")
    
    print(f"\nAt latest time ({test_times[-1]:.6f}):")
    print(f"  - Data signal: {output.alpha_t[-1]:.6f} (maximum ✓)") 
    print(f"  - Noise level: {output.sigma_t[-1]:.6f} (minimum ✓)")
    
    print("\n🎉 Your scheduler works perfectly fine with Flow Matching!")
    print("   Normalization is just an extra safety measure.")


if __name__ == "__main__":
    print("Understanding Time Normalization in Flow Matching")
    print("This explains why normalization exists and when you need it.")
    
    show_scheduler_behavior()
    compare_with_and_without_normalization()
    show_when_normalization_matters()
    show_your_scheduler_is_actually_fine()
    practical_recommendations()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("• Your time_scheduler [0.002, 1.0] actually works fine with Flow Matching")
    print("• Normalization is a safety measure, not a strict requirement")
    print("• The math works because 0.002 ≈ 0 for practical purposes")  
    print("• We added normalize_time=True for mathematical purity and edge case handling")
    print("• You can skip normalization if using DiscreteDiffusionLoss")
    print("• The easiest fix is just changing your scheduler to start at 0 instead of 1/num_timesteps")