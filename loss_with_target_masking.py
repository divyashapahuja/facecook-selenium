"""
Modified Flow Matching loss that uses your target preparation logic internally.
This gives you the same masking behavior as your original training.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from dataclasses import dataclass, field

# Copy the necessary classes from loss_final.py
@dataclass
class SchedulerOutput:
    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})

class TimeSchedulerWrapper:
    def __init__(self, time_scheduler: Tensor):
        self.time_scheduler = time_scheduler
        
    def __call__(self, timesteps: Tensor) -> SchedulerOutput:
        t = self.time_scheduler[timesteps]
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

class MixtureDiscreteProbPath:
    def __init__(self, time_scheduler: Tensor):
        self.scheduler = TimeSchedulerWrapper(time_scheduler)


class MixturePathGeneralizedKL_WithTargetMasking(_Loss):
    """
    Flow Matching loss that internally uses your target preparation logic.
    This gives you the same masking behavior as F.cross_entropy with ignore_index=-100.
    """

    def __init__(self, time_scheduler: Tensor, mask_token_id: int, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = MixtureDiscreteProbPath(time_scheduler)
        self.mask_token_id = mask_token_id

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, timesteps: Tensor, 
                attention_mask: Tensor) -> Tensor:
        """
        Args:
            logits: Model predictions, shape (batch, seq_len, vocab_size)
            x_1: Clean tokens, shape (batch, seq_len)  
            x_t: Noised tokens, shape (batch, seq_len)
            timesteps: Timestep indices, shape (batch,)
            attention_mask: Attention mask, shape (batch, seq_len)
        """
        x_1_shape = x_1.shape

        # Extract x_1 value of log(p_{1|t}(x|x_t))
        log_p_1t = torch.log_softmax(logits, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1))
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        # Extract x_t value of p_{1|t}(x|x_t)
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1))
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(timesteps)

        jump_coefficient = (
            scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)
        )[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
        
        delta_x1_xt = (x_t == x_1).to(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        # YOUR TARGET MASKING LOGIC (applied to loss instead of target)
        # Create mask for positions we want to calculate loss on
        valid_mask = torch.ones_like(x_1, dtype=torch.bool)
        
        # Mask out positions that weren't noised (equivalent to your target != -100 logic)
        valid_mask = valid_mask & (x_t != x_1)  # Only calculate loss where tokens were actually masked
        
        # Mask out padded positions
        valid_mask = valid_mask & attention_mask.bool()
        
        # Apply the mask to loss (equivalent to your ignore_index=-100)
        loss = loss * valid_mask.to(loss.dtype)

        if self.reduction == "mean":
            # Average only over valid positions (like your original approach)
            return torch.sum(loss) / torch.sum(valid_mask.float())
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")


class MixturePathGeneralizedKL_CrossEntropyStyle(_Loss):
    """
    Alternative: Flow Matching loss that mimics F.cross_entropy behavior exactly.
    This version uses your exact target preparation logic.
    """

    def __init__(self, time_scheduler: Tensor, mask_token_id: int, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = MixtureDiscreteProbPath(time_scheduler)
        self.mask_token_id = mask_token_id

    def prepare_target_like_original(self, input_ids: Tensor, noised_input_ids: Tensor, 
                                   attention_mask: Tensor) -> Tensor:
        """
        YOUR EXACT target preparation logic from the original training code.
        """
        target = input_ids.clone()
        
        # Only calculate loss on tokens that were masked
        target[noised_input_ids != self.mask_token_id] = -100
        target[attention_mask == 0] = -100
        
        return target

    def forward(self, logits: Tensor, input_ids: Tensor, noised_input_ids: Tensor, 
                timesteps: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Args:
            logits: Model predictions, shape (batch, seq_len, vocab_size)
            input_ids: Clean tokens, shape (batch, seq_len)
            noised_input_ids: Noised tokens, shape (batch, seq_len)  
            timesteps: Timestep indices, shape (batch,)
            attention_mask: Attention mask, shape (batch, seq_len)
        """
        # Prepare target using YOUR exact logic
        target = self.prepare_target_like_original(input_ids, noised_input_ids, attention_mask)
        
        # Create mask for valid positions (where target != -100)
        valid_mask = (target != -100)
        
        # Only calculate Flow Matching loss on valid positions
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Extract valid positions
        valid_logits = logits[valid_mask]  # Shape: (num_valid, vocab_size)
        valid_input_ids = input_ids[valid_mask]  # Shape: (num_valid,)
        valid_noised_ids = noised_input_ids[valid_mask]  # Shape: (num_valid,)
        
        # Get timesteps for valid positions (broadcast to match valid positions)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1).expand_as(input_ids)[valid_mask]
        valid_timesteps = timesteps[batch_indices]
        
        # Calculate Flow Matching loss only on valid positions
        x_1_shape = valid_input_ids.shape

        # Extract x_1 value of log(p_{1|t}(x|x_t))
        log_p_1t = torch.log_softmax(valid_logits, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=valid_input_ids.unsqueeze(-1))
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        # Extract x_t value of p_{1|t}(x|x_t)
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=valid_noised_ids.unsqueeze(-1))
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(valid_timesteps)

        jump_coefficient = scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)
        
        delta_x1_xt = (valid_noised_ids == valid_input_ids).to(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            # Return full-size tensor with zeros for invalid positions
            full_loss = torch.zeros_like(input_ids, dtype=loss.dtype)
            full_loss[valid_mask] = loss
            return full_loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")


# USAGE EXAMPLES:

def example_usage_option_1():
    """
    Option 1: Modified loss that applies your masking logic internally
    """
    # Your setup (unchanged)
    time_scheduler = torch.linspace(1/512, 1, steps=512)
    loss_fn = MixturePathGeneralizedKL_WithTargetMasking(
        time_scheduler, 
        mask_token_id=50256,  # Your mask token ID
        reduction="mean"
    )
    
    # In training loop:
    # ... your existing code ...
    
    # Flow Matching loss with internal target masking
    loss = loss_fn(
        logits=logits,
        x_1=input_ids,           # Clean tokens
        x_t=noised_input_ids,    # Noised tokens (actual token IDs, not -100!)
        timesteps=timesteps,
        attention_mask=attention_mask
    )


def example_usage_option_2():
    """
    Option 2: Loss that uses your exact target preparation logic
    """
    # Your setup (unchanged)
    time_scheduler = torch.linspace(1/512, 1, steps=512)
    loss_fn = MixturePathGeneralizedKL_CrossEntropyStyle(
        time_scheduler,
        mask_token_id=50256,  # Your mask token ID  
        reduction="mean"
    )
    
    # In training loop:
    # ... your existing code ...
    
    # Flow Matching loss with your exact target preparation
    loss = loss_fn(
        logits=logits,
        input_ids=input_ids,         # Clean tokens
        noised_input_ids=noised_input_ids,  # Noised tokens
        timesteps=timesteps,
        attention_mask=attention_mask
    )


if __name__ == "__main__":
    print("Two approaches to use your target masking logic with Flow Matching:")
    print()
    print("Option 1: MixturePathGeneralizedKL_WithTargetMasking")
    print("- Applies your masking logic to the loss tensor")
    print("- Still passes actual token IDs as x_t")
    print("- Most similar to current Flow Matching approach")
    print()
    print("Option 2: MixturePathGeneralizedKL_CrossEntropyStyle") 
    print("- Uses your EXACT target preparation logic")
    print("- Only calculates loss on valid positions")
    print("- Most similar to your original F.cross_entropy approach")
    print()
    print("Both give you the same masking behavior as your original training!")