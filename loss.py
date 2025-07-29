import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math


class GeneralizedKL(nn.Module):
    """
    Generalized KL divergence loss adapted from Facebook Research flow_matching repository
    for discrete flow language models.
    
    This implements the generalized flow matching loss that accounts for the probability path
    interpolation between source and target distributions in discrete space.
    """
    
    def __init__(self, 
                 alpha_t: torch.Tensor,
                 sigma_t: torch.Tensor, 
                 d_alpha_t: torch.Tensor,
                 d_sigma_t: torch.Tensor,
                 reduction: str = "mean"):
        """
        Args:
            alpha_t: Alpha coefficient at time t (controls mixing with data)
            sigma_t: Sigma coefficient at time t (controls noise level)  
            d_alpha_t: Derivative of alpha with respect to time
            d_sigma_t: Derivative of sigma with respect to time
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha_t = alpha_t
        self.sigma_t = sigma_t
        self.d_alpha_t = d_alpha_t
        self.d_sigma_t = d_sigma_t
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized KL loss for discrete flow matching.
        
        Args:
            input: Model predictions (logits) of shape [B, V, L] where B=batch, V=vocab, L=length
            target: Target token ids of shape [B, L]
            
        Returns:
            Loss tensor
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(input, dim=1)  # [B, V, L]
        
        # Standard cross entropy component
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=-100)  # [B, L]
        
        # Flow matching correction terms
        # These terms account for the probability path interpolation
        batch_size, vocab_size, seq_len = input.shape
        
        # Compute the flow matching correction
        # This is based on the generalized loss from the flow matching paper
        probs = F.softmax(input, dim=1)  # [B, V, L]
        
        # Create one-hot encoding for targets
        target_valid = target != -100  # [B, L]
        target_clamped = torch.clamp(target, min=0)  # Replace -100 with 0 for indexing
        target_onehot = F.one_hot(target_clamped, num_classes=vocab_size).float()  # [B, L, V]
        target_onehot = target_onehot.transpose(1, 2)  # [B, V, L]
        
        # Apply validity mask
        target_onehot = target_onehot * target_valid.unsqueeze(1)  # [B, V, L]
        
        # Flow matching terms based on the probability path
        # α(t) * data + σ(t) * noise interpolation
        alpha_term = self.alpha_t * torch.sum(probs * target_onehot, dim=1)  # [B, L]
        sigma_term = self.sigma_t * torch.sum(probs * (1 - target_onehot), dim=1)  # [B, L]
        
        # Derivative terms for the velocity field
        d_alpha_term = self.d_alpha_t * torch.sum(probs * target_onehot, dim=1)  # [B, L]
        d_sigma_term = self.d_sigma_t * torch.sum(probs * (1 - target_onehot), dim=1)  # [B, L]
        
        # Combine terms - this is the generalized flow matching loss
        flow_correction = d_alpha_term + d_sigma_term - (alpha_term + sigma_term)
        
        # Total loss combines cross entropy with flow matching correction
        total_loss = ce_loss + flow_correction
        
        # Apply validity mask
        total_loss = total_loss * target_valid
        
        # Apply reduction
        if self.reduction == "mean":
            return total_loss.sum() / target_valid.sum().clamp(min=1)
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


class MixturePathGeneralizedKL(nn.Module):
    """
    Mixture path generalized KL loss for discrete flow matching.
    
    This implements a mixture of probability paths with different schedulers,
    allowing for more flexible interpolation between source and target distributions.
    """
    
    def __init__(self, 
                 alpha_t: torch.Tensor,
                 sigma_t: torch.Tensor,
                 d_alpha_t: torch.Tensor, 
                 d_sigma_t: torch.Tensor,
                 mixture_weights: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        """
        Args:
            alpha_t: Alpha coefficients for different paths [num_paths] or [batch_size, num_paths]
            sigma_t: Sigma coefficients for different paths [num_paths] or [batch_size, num_paths]
            d_alpha_t: Alpha derivatives [num_paths] or [batch_size, num_paths]
            d_sigma_t: Sigma derivatives [num_paths] or [batch_size, num_paths]
            mixture_weights: Weights for mixing different paths [num_paths]
            reduction: Reduction method
        """
        super().__init__()
        self.alpha_t = alpha_t
        self.sigma_t = sigma_t
        self.d_alpha_t = d_alpha_t
        self.d_sigma_t = d_sigma_t
        
        if mixture_weights is None:
            num_paths = alpha_t.shape[-1] if alpha_t.dim() > 0 else 1
            mixture_weights = torch.ones(num_paths) / num_paths
        self.mixture_weights = mixture_weights
        self.reduction = reduction
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute mixture path generalized KL loss.
        
        Args:
            input: Model predictions [B, V, L]
            target: Target tokens [B, L]
            
        Returns:
            Loss tensor
        """
        batch_size, vocab_size, seq_len = input.shape
        
        # Handle different path dimensions
        if self.alpha_t.dim() == 1:
            # Single set of coefficients for all samples
            num_paths = len(self.alpha_t)
            alpha_t = self.alpha_t.unsqueeze(0).expand(batch_size, -1)  # [B, num_paths]
            sigma_t = self.sigma_t.unsqueeze(0).expand(batch_size, -1)
            d_alpha_t = self.d_alpha_t.unsqueeze(0).expand(batch_size, -1)
            d_sigma_t = self.d_sigma_t.unsqueeze(0).expand(batch_size, -1)
        else:
            # Different coefficients per sample
            alpha_t = self.alpha_t
            sigma_t = self.sigma_t
            d_alpha_t = self.d_alpha_t
            d_sigma_t = self.d_sigma_t
            num_paths = alpha_t.shape[1]
        
        # Compute loss for each path
        path_losses = []
        
        for path_idx in range(num_paths):
            # Create generalized KL loss for this path
            path_loss_fn = GeneralizedKL(
                alpha_t=alpha_t[:, path_idx] if alpha_t.dim() > 1 else alpha_t[path_idx],
                sigma_t=sigma_t[:, path_idx] if sigma_t.dim() > 1 else sigma_t[path_idx],
                d_alpha_t=d_alpha_t[:, path_idx] if d_alpha_t.dim() > 1 else d_alpha_t[path_idx],
                d_sigma_t=d_sigma_t[:, path_idx] if d_sigma_t.dim() > 1 else d_sigma_t[path_idx],
                reduction='none'
            )
            
            path_loss = path_loss_fn(input, target)  # [B, L]
            path_losses.append(path_loss)
        
        # Stack and weight the losses
        path_losses = torch.stack(path_losses, dim=0)  # [num_paths, B, L]
        weights = self.mixture_weights.view(-1, 1, 1).to(path_losses.device)  # [num_paths, 1, 1]
        
        # Weighted mixture of path losses
        mixed_loss = torch.sum(weights * path_losses, dim=0)  # [B, L]
        
        # Apply reduction
        target_valid = target != -100
        mixed_loss = mixed_loss * target_valid
        
        if self.reduction == "mean":
            return mixed_loss.sum() / target_valid.sum().clamp(min=1)
        elif self.reduction == "sum":
            return mixed_loss.sum()
        else:
            return mixed_loss


class MixtureDiscreteProbPath(nn.Module):
    """
    Discrete probability path for flow matching in token space.
    
    This implements discrete flow matching where we interpolate between
    noise tokens and data tokens using learnable probability paths.
    """
    
    def __init__(self,
                 vocab_size: int,
                 mask_token_id: int,
                 time_scheduler: Callable[[torch.Tensor], torch.Tensor],
                 reduction: str = "mean"):
        """
        Args:
            vocab_size: Size of vocabulary
            mask_token_id: ID of the mask token used for noising
            time_scheduler: Function that maps time to interpolation coefficients
            reduction: Reduction method
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.time_scheduler = time_scheduler
        self.reduction = reduction
        
    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete probability path loss.
        
        Args:
            input: Model predictions [B, V, L]
            target: Target tokens [B, L] 
            timesteps: Time steps for each sample [B]
            
        Returns:
            Loss tensor
        """
        batch_size, vocab_size, seq_len = input.shape
        
        # Get interpolation coefficients from time scheduler
        t_coeffs = self.time_scheduler(timesteps)  # [B]
        
        # Compute probability interpolation
        log_probs = F.log_softmax(input, dim=1)  # [B, V, L]
        probs = torch.exp(log_probs)
        
        # Create target distribution (one-hot for data tokens)
        target_valid = target != -100
        target_clamped = torch.clamp(target, min=0)
        target_onehot = F.one_hot(target_clamped, num_classes=vocab_size).float()  # [B, L, V]
        target_onehot = target_onehot.transpose(1, 2)  # [B, V, L]
        target_onehot = target_onehot * target_valid.unsqueeze(1)
        
        # Create noise distribution (uniform over non-mask tokens or concentrated on mask)  
        noise_dist = torch.ones_like(probs) / vocab_size
        noise_dist[:, self.mask_token_id, :] = 1.0  # Concentrate on mask token
        noise_dist = noise_dist / noise_dist.sum(dim=1, keepdim=True)
        
        # Interpolate between noise and data distributions
        t_coeffs = t_coeffs.view(-1, 1, 1)  # [B, 1, 1]
        interpolated_dist = (1 - t_coeffs) * noise_dist + t_coeffs * target_onehot
        
        # KL divergence between predicted and interpolated distributions
        kl_loss = F.kl_div(log_probs, interpolated_dist, reduction='none')  # [B, V, L]
        kl_loss = kl_loss.sum(dim=1)  # [B, L]
        
        # Apply validity mask
        kl_loss = kl_loss * target_valid
        
        # Apply reduction
        if self.reduction == "mean":
            return kl_loss.sum() / target_valid.sum().clamp(min=1)
        elif self.reduction == "sum":
            return kl_loss.sum()
        else:
            return kl_loss


def create_flow_matching_schedulers(num_timesteps: int, 
                                   device: torch.device,
                                   scheduler_type: str = "linear") -> tuple:
    """
    Create time-dependent coefficients for flow matching.
    
    Args:
        num_timesteps: Number of discrete time steps
        device: Device to place tensors on
        scheduler_type: Type of scheduler ("linear", "cosine", "exponential")
        
    Returns:
        Tuple of (alpha_t, sigma_t, d_alpha_t, d_sigma_t)
    """
    t = torch.linspace(0, 1, num_timesteps, device=device)
    
    if scheduler_type == "linear":
        # Linear interpolation: α(t) = t, σ(t) = 1-t
        alpha_t = t
        sigma_t = 1 - t
        d_alpha_t = torch.ones_like(t)
        d_sigma_t = -torch.ones_like(t)
        
    elif scheduler_type == "cosine":
        # Cosine schedule for smoother interpolation
        alpha_t = 0.5 * (1 + torch.cos(math.pi * (1 - t)))
        sigma_t = 0.5 * (1 + torch.cos(math.pi * t))
        d_alpha_t = 0.5 * math.pi * torch.sin(math.pi * (1 - t))
        d_sigma_t = -0.5 * math.pi * torch.sin(math.pi * t)
        
    elif scheduler_type == "exponential":
        # Exponential schedule
        alpha_t = torch.exp(-2 * (1 - t))
        sigma_t = torch.exp(-2 * t)
        d_alpha_t = 2 * torch.exp(-2 * (1 - t))
        d_sigma_t = -2 * torch.exp(-2 * t)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return alpha_t, sigma_t, d_alpha_t, d_sigma_t


# Helper function to create the loss function used in training
def create_mixture_path_loss(num_timesteps: int,
                           device: torch.device,
                           scheduler_type: str = "linear",
                           reduction: str = "mean") -> MixturePathGeneralizedKL:
    """
    Create a MixturePathGeneralizedKL loss function with appropriate schedulers.
    
    Args:
        num_timesteps: Number of time steps
        device: Device for tensors
        scheduler_type: Type of time scheduler
        reduction: Loss reduction method
        
    Returns:
        Configured loss function
    """
    alpha_t, sigma_t, d_alpha_t, d_sigma_t = create_flow_matching_schedulers(
        num_timesteps, device, scheduler_type
    )
    
    return MixturePathGeneralizedKL(
        alpha_t=alpha_t,
        sigma_t=sigma_t, 
        d_alpha_t=d_alpha_t,
        d_sigma_t=d_sigma_t,
        reduction=reduction
    )