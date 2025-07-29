# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from dataclasses import dataclass, field


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).
    """
    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class TimeSchedulerWrapper:
    """
    Wrapper to make your existing time_scheduler work with Flow Matching.
    This converts your time_scheduler values to the format expected by Flow Matching.
    """
    
    def __init__(self, time_scheduler: Tensor):
        """
        Args:
            time_scheduler: Your existing time scheduler tensor, e.g., 
                          torch.linspace(1/num_timesteps, 1, steps=num_timesteps)
        """
        self.time_scheduler = time_scheduler
        
    def __call__(self, timesteps: Tensor) -> SchedulerOutput:
        """
        Convert timestep indices to Flow Matching scheduler output.
        
        Args:
            timesteps: Tensor of timestep indices, shape (batch_size,)
            
        Returns:
            SchedulerOutput with alpha_t, sigma_t, and derivatives
        """
        # Get time values from your scheduler
        t = self.time_scheduler[timesteps]
        
        # Flow Matching linear scheduler: alpha_t = t, sigma_t = 1 - t
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )


class MixtureDiscreteProbPath:
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, 
    determined by the scheduler, when it flips to the target data point :math:`X_1`.

    Args:
        time_scheduler (Tensor): Your existing time scheduler tensor.
    """

    def __init__(self, time_scheduler: Tensor):
        self.scheduler = TimeSchedulerWrapper(time_scheduler)


class MixturePathGeneralizedKL(_Loss):
    r"""A generalized KL loss for discrete flow matching adapted for diffusion training.
    
    This class measures the generalized KL of a discrete flow model w.r.t. a probability 
    path given by ``path``. It's adapted to work with your existing time_scheduler.

    Args:
        time_scheduler (Tensor): Your existing time scheduler tensor.
        reduction (str, optional): Specify the reduction to apply to the output 
            ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to 'mean'.
    """

    def __init__(self, time_scheduler: Tensor, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = MixtureDiscreteProbPath(time_scheduler)

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, timesteps: Tensor, 
                attention_mask: Tensor = None) -> Tensor:
        r"""Evaluates the generalized KL loss adapted for diffusion training.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(``logits``) = p_{1|t}(x|x_t)), 
                           shape (batch, seq_len, vocab_size).
            x_1 (Tensor): target data point (clean tokens), shape (batch, seq_len).
            x_t (Tensor): conditional sample at x_t ~ p_t(·|x_1) (noised tokens), 
                         shape (batch, seq_len).
            timesteps (Tensor): timestep indices, shape (batch).
            attention_mask (Tensor, optional): attention mask, shape (batch, seq_len).

        Returns:
            Tensor: Generalized KL loss.
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

        # Apply attention mask if provided
        if attention_mask is not None:
            loss = loss * attention_mask.to(loss.dtype)

        if self.reduction == "mean":
            if attention_mask is not None:
                return torch.sum(loss) / torch.sum(attention_mask)
            else:
                return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")


class DiscreteDiffusionLoss(_Loss):
    """
    A simplified discrete diffusion loss that can be used as an alternative to the 
    Flow Matching loss. This matches the training setup in your code more directly.
    """
    
    def __init__(self, mask_token_id: int, reduction: str = "mean"):
        super().__init__(None, None, reduction)
        self.mask_token_id = mask_token_id
    
    def forward(self, logits: Tensor, target: Tensor, attention_mask: Tensor = None) -> Tensor:
        r"""
        Args:
            logits (Tensor): Model predictions, shape (batch, seq_len, vocab_size)
            target (Tensor): Target tokens with -100 for ignored positions, shape (batch, seq_len)
            attention_mask (Tensor, optional): Attention mask, shape (batch, seq_len)
        """
        # Use cross-entropy loss with ignore_index=-100
        loss = F.cross_entropy(
            input=logits.transpose(-1, -2),  # CE expects (batch, vocab_size, seq_len)
            target=target,
            reduction="none",
            ignore_index=-100
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            loss = loss * attention_mask.to(loss.dtype)
            
        if self.reduction == "mean":
            if attention_mask is not None:
                return torch.sum(loss) / torch.sum(attention_mask)
            else:
                # Only average over non-ignored positions
                valid_positions = (target != -100).float()
                return torch.sum(loss) / torch.sum(valid_positions)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")