"""
Debug script to investigate high Flow Matching loss values.
This will help identify scaling issues and compare with cross-entropy loss.
"""

import torch
import torch.nn.functional as F
from loss_properly_fixed import MixturePathGeneralizedKL, DiscreteDiffusionLoss

def debug_loss_values():
    """Debug and compare different loss implementations."""
    
    print("🔍 Debugging Flow Matching Loss Values")
    print("=" * 50)
    
    # Create realistic test data
    batch_size, seq_len, vocab_size = 4, 32, 1000
    num_timesteps = 128
    mask_token_id = 999
    
    # Create time scheduler
    time_scheduler = torch.linspace(1/num_timesteps, 1, steps=num_timesteps)
    
    # Create test tensors
    input_ids = torch.randint(0, vocab_size-1, (batch_size, seq_len))  # Avoid mask token
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Simulate some padding
    attention_mask[:, -5:] = 0  # Last 5 tokens are padding
    
    # Create realistic logits (not too random)
    logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1  # Smaller variance
    
    # Create noised input (simulate your model._create_noise_ids)
    timesteps = torch.randint(0, num_timesteps, (batch_size,))
    time_t = time_scheduler[timesteps]
    
    # Simulate noise: randomly replace some tokens with mask_token_id
    noised_input_ids = input_ids.clone()
    noise_mask = torch.rand(batch_size, seq_len) < time_t.unsqueeze(1)  # More noise at higher t
    noised_input_ids[noise_mask] = mask_token_id
    
    print(f"📊 Test Setup:")
    print(f"   Batch size: {batch_size}, Seq len: {seq_len}, Vocab size: {vocab_size}")
    print(f"   Time values (t): {time_t}")
    print(f"   Noise ratio: {noise_mask.float().mean():.3f}")
    print()
    
    # 1. Original Cross-Entropy Loss (your current approach)
    print("1️⃣ Original Cross-Entropy Loss:")
    target = input_ids.clone()
    target[noised_input_ids != mask_token_id] = -100  # Only predict masked tokens
    target[attention_mask == 0] = -100  # Ignore padding
    
    ce_loss = F.cross_entropy(
        input=logits.transpose(-1, -2),
        target=target,
        reduction="mean",
        ignore_index=-100
    )
    print(f"   Cross-entropy loss: {ce_loss.item():.4f}")
    
    # 2. Discrete Diffusion Loss (improved CE)
    print("\n2️⃣ Discrete Diffusion Loss:")
    discrete_loss_fn = DiscreteDiffusionLoss(mask_token_id=mask_token_id, reduction="mean")
    discrete_loss = discrete_loss_fn(logits=logits, target=target, attention_mask=attention_mask)
    print(f"   Discrete diffusion loss: {discrete_loss.item():.4f}")
    
    # 3. Flow Matching Loss (current implementation)
    print("\n3️⃣ Flow Matching Loss (current):")
    flow_loss_fn = MixturePathGeneralizedKL(time_scheduler, reduction="mean")
    flow_loss = flow_loss_fn(
        logits=logits,
        x_1=input_ids,
        x_t=noised_input_ids,
        t=time_t,
        attention_mask=attention_mask
    )
    print(f"   Flow Matching loss: {flow_loss.item():.4f}")
    
    # 4. Debug Flow Matching components
    print("\n🔍 Flow Matching Loss Components:")
    
    # Manually compute Flow Matching loss to debug
    x_1_shape = input_ids.shape
    
    # Extract probabilities
    log_p_1t = torch.log_softmax(logits, dim=-1)
    log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=input_ids.unsqueeze(-1))
    log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)
    
    p_1t = torch.exp(log_p_1t)
    p_1t_xt = torch.gather(p_1t, dim=-1, index=noised_input_ids.unsqueeze(-1))
    p_1t_xt = p_1t_xt.view(*x_1_shape)
    
    # Scheduler output
    alpha_t = time_t
    d_alpha_t = torch.ones_like(time_t)
    
    # Jump coefficient
    jump_coefficient = (d_alpha_t / (1 - alpha_t))[(...,) + (None,) * (input_ids.dim() - 1)]
    jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
    
    # Delta function
    delta_x1_xt = (noised_input_ids == input_ids).to(logits.dtype)
    
    # Loss components
    loss_component = -jump_coefficient * (
        p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
    )
    
    print(f"   Jump coefficient range: [{jump_coefficient.min():.3f}, {jump_coefficient.max():.3f}]")
    print(f"   p_1t_xt range: [{p_1t_xt.min():.6f}, {p_1t_xt.max():.6f}]")
    print(f"   log_p_1t_x1 range: [{log_p_1t_x1.min():.3f}, {log_p_1t_x1.max():.3f}]")
    print(f"   delta_x1_xt mean: {delta_x1_xt.float().mean():.3f}")
    print(f"   Loss component range: [{loss_component.min():.3f}, {loss_component.max():.3f}]")
    
    # Apply attention mask
    if attention_mask is not None:
        loss_component = loss_component * attention_mask.to(loss_component.dtype)
        final_loss = torch.sum(loss_component) / torch.sum(attention_mask)
    else:
        final_loss = torch.mean(loss_component)
    
    print(f"   Manual FM loss: {final_loss.item():.4f}")
    
    # 5. Potential fixes
    print("\n🛠️ Potential Issues & Fixes:")
    
    # Check if jump coefficient is too large
    max_jump = jump_coefficient.max().item()
    if max_jump > 10:
        print(f"   ⚠️  Jump coefficient too large: {max_jump:.3f}")
        print(f"       This happens when (1 - alpha_t) is small (t close to 1)")
        print(f"       Consider clipping or using a different scheduler")
    
    # Check if we're only calculating loss on masked positions
    masked_positions = (noised_input_ids == mask_token_id).float().mean()
    if masked_positions < 0.1:
        print(f"   ⚠️  Very few masked positions: {masked_positions:.3f}")
        print(f"       Flow Matching loss might be unstable with few masked tokens")
    
    # Check time values
    if time_t.min() > 0.9:
        print(f"   ⚠️  Time values very high: [{time_t.min():.3f}, {time_t.max():.3f}]")
        print(f"       High t values lead to large jump coefficients")
    
    return {
        'cross_entropy': ce_loss.item(),
        'discrete_diffusion': discrete_loss.item(),
        'flow_matching': flow_loss.item(),
        'jump_coefficient_max': max_jump,
        'masked_ratio': masked_positions
    }


def suggest_fixes():
    """Suggest potential fixes for high loss values."""
    print("\n💡 Suggested Fixes:")
    print("=" * 50)
    
    print("1️⃣ **Clip Jump Coefficient**: Prevent extreme values")
    print("   jump_coefficient = torch.clamp(jump_coefficient, max=10.0)")
    
    print("\n2️⃣ **Scale Loss**: Reduce overall magnitude")
    print("   loss = loss * 0.1  # or another scaling factor")
    
    print("\n3️⃣ **Only Calculate Loss on Masked Positions**: Like your original")
    print("   Apply loss only where noised_input_ids == mask_token_id")
    
    print("\n4️⃣ **Use Different Time Range**: Avoid t close to 1")
    print("   time_scheduler = torch.linspace(0.1, 0.9, steps=num_timesteps)")
    
    print("\n5️⃣ **Gradual Transition**: Start with discrete loss, then FM")
    print("   Mix losses: (1-alpha) * discrete_loss + alpha * fm_loss")


def create_fixed_loss():
    """Create a more stable Flow Matching loss."""
    print("\n🔧 Creating Improved Flow Matching Loss:")
    print("=" * 50)
    
    code = '''
class ImprovedMixturePathGeneralizedKL(MixturePathGeneralizedKL):
    def __init__(self, time_scheduler: Tensor, reduction: str = "mean", 
                 max_jump_coeff: float = 10.0, loss_scale: float = 1.0):
        super().__init__(time_scheduler, reduction)
        self.max_jump_coeff = max_jump_coeff
        self.loss_scale = loss_scale
    
    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor, 
                attention_mask: Tensor = None) -> Tensor:
        # ... same as before until jump_coefficient calculation ...
        
        # Clip jump coefficient to prevent extreme values
        jump_coefficient = torch.clamp(jump_coefficient, max=self.max_jump_coeff)
        
        # Only calculate loss on masked positions (like your original approach)
        mask_positions = (x_t != x_1).float()
        loss = loss * mask_positions
        
        # Scale the loss
        loss = loss * self.loss_scale
        
        # ... rest same as before ...
    '''
    
    print(code)


if __name__ == "__main__":
    results = debug_loss_values()
    suggest_fixes()
    create_fixed_loss()
    
    print(f"\n📈 Summary:")
    print(f"   Cross-entropy loss: {results['cross_entropy']:.4f}")
    print(f"   Flow Matching loss: {results['flow_matching']:.4f}")
    print(f"   Ratio (FM/CE): {results['flow_matching']/results['cross_entropy']:.1f}x")
    
    if results['flow_matching'] > results['cross_entropy'] * 5:
        print(f"   🚨 Flow Matching loss is {results['flow_matching']/results['cross_entropy']:.1f}x higher!")
        print(f"      Consider applying the suggested fixes above.")