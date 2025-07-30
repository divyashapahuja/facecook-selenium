# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union

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


@dataclass
class DiscretePathSample:
    """
    Represents a sample of a conditional-flow generated discrete probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample  :math:`t`.
        x_t (Tensor): the sample along the path  :math:`X_t \sim p_t`.
    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."}
    )


def unsqueeze_to_match(source: Tensor, target: Tensor, how: str = "suffix") -> Tensor:
    """
    Unsqueeze the source tensor to match the dimensionality of the target tensor.

    Args:
        source (Tensor): The source tensor to be unsqueezed.
        target (Tensor): The target tensor to match the dimensionality of.
        how (str, optional): Whether to unsqueeze the source tensor at the beginning
            ("prefix") or end ("suffix"). Defaults to "suffix".

    Returns:
        Tensor: The unsqueezed source tensor.
    """
    assert (
        how == "prefix" or how == "suffix"
    ), f"{how} is not supported, only 'prefix' and 'suffix' are supported."

    dim_diff = target.dim() - source.dim()

    for _ in range(dim_diff):
        if how == "prefix":
            source = source.unsqueeze(0)
        elif how == "suffix":
            source = source.unsqueeze(-1)

    return source


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)


class MixtureDiscreteProbPath:
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, 
    determined by the scheduler, when it flips to the target data point :math:`X_1`.

    Args:
        time_scheduler (Tensor): Your existing time scheduler tensor.
    """

    def __init__(self, time_scheduler: Tensor):
        # Store the time_scheduler for reference, but we don't need it for the scheduler method
        self.time_scheduler = time_scheduler

    def scheduler(self, t: Tensor) -> SchedulerOutput:
        r"""
        Create scheduler output from time values t.
        
        Args: 
            t: Tensor of time values, shape (batch_size,)

        Returns: 
            SchedulerOutput with alpha_t, sigma_t, d_alpha_t, and d_sigma_t
        """
        # For Flow Matching linear scheduler: alpha_t = t, sigma_t = 1 - t
        return SchedulerOutput(
            alpha_t=t, 
            sigma_t=1 - t, 
            d_alpha_t=torch.ones_like(t), 
            d_sigma_t=-torch.ones_like(t)
        )


class MixturePathGeneralizedKL(_Loss):
    r"""A STABLE generalized KL loss for discrete flow matching.
    
    This version includes several stability improvements:
    - Clips jump coefficients to prevent extreme values
    - Only calculates loss on masked positions (like your original approach)
    - Optional loss scaling
    - Better handling of edge cases

    Args:
        time_scheduler (Tensor): Your existing time scheduler tensor.
        reduction (str, optional): Specify the reduction to apply to the output 
            ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to 'mean'.
        max_jump_coeff (float, optional): Maximum jump coefficient value. Defaults to 10.0.
        loss_scale (float, optional): Scale factor for the loss. Defaults to 1.0.
        mask_token_id (int, optional): Only calculate loss on positions with this token. 
            If None, calculate on all positions. Defaults to None.
    """

    def __init__(self, time_scheduler: Tensor, reduction: str = "mean", 
                 max_jump_coeff: float = 10.0, loss_scale: float = 1.0,
                 mask_token_id: int = None) -> None:
        super().__init__(None, None, reduction)
        self.path = MixtureDiscreteProbPath(time_scheduler)
        self.max_jump_coeff = max_jump_coeff
        self.loss_scale = loss_scale
        self.mask_token_id = mask_token_id

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor, 
                attention_mask: Tensor = None) -> Tensor:
        r"""Evaluates the generalized KL loss with stability improvements.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(``logits``) :math:`=p_{1|t}(x|x_t)`), shape (batch, d, K).
            x_1 (Tensor): target data point :math:`x_1 \sim q`, shape (batch, d).
            x_t (Tensor): conditional sample at :math:`x_t \sim p_t(\cdot|x_1)`, shape (batch, d).
            t (Tensor): times in :math:`[0,1]`, shape (batch).
            attention_mask (Tensor, optional): attention mask, shape (batch, d).

        Returns:
            Tensor: Generalized KL loss.
        """
        x_1_shape = x_1.shape

        # extract x_1 value of log(p_{1|t}(x|x_t)).
        log_p_1t = torch.log_softmax(logits, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1))
        log_p_1t_x1 = log_p_1t_x1.view(*x_1_shape)

        # extract x_t value of p_{1|t}(x|x_t).
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1))
        p_1t_xt = p_1t_xt.view(*x_1_shape)

        scheduler_output = self.path.scheduler(t)

        # STABILITY FIX 1: Clip jump coefficient to prevent extreme values
        jump_coefficient = scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)
        jump_coefficient = torch.clamp(jump_coefficient, max=self.max_jump_coeff)
        
        jump_coefficient = jump_coefficient[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
        
        delta_x1_xt = (x_t == x_1).to(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        # STABILITY FIX 2: Only calculate loss on masked positions (like your original)
        if self.mask_token_id is not None:
            mask_positions = (x_t == self.mask_token_id).to(loss.dtype)
            loss = loss * mask_positions

        # STABILITY FIX 3: Apply loss scaling
        loss = loss * self.loss_scale

        # Apply attention mask if provided
        if attention_mask is not None:
            loss = loss * attention_mask.to(loss.dtype)

        if self.reduction == "mean":
            if attention_mask is not None:
                valid_tokens = torch.sum(attention_mask)
                if self.mask_token_id is not None:
                    # Only count masked positions in denominator
                    valid_tokens = torch.sum(attention_mask * (x_t == self.mask_token_id).float())
                return torch.sum(loss) / torch.clamp(valid_tokens, min=1.0)  # Avoid division by zero
            else:
                if self.mask_token_id is not None:
                    # Only count masked positions in denominator
                    valid_tokens = torch.sum((x_t == self.mask_token_id).float())
                    return torch.sum(loss) / torch.clamp(valid_tokens, min=1.0)
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


# Convenience function to create a stable Flow Matching loss
def create_stable_flow_matching_loss(time_scheduler: Tensor, mask_token_id: int, 
                                   loss_scale: float = 0.1) -> MixturePathGeneralizedKL:
    """
    Create a stable Flow Matching loss with recommended settings.
    
    Args:
        time_scheduler: Your existing time scheduler
        mask_token_id: Token ID used for masking
        loss_scale: Scale factor to reduce loss magnitude (default: 0.1)
    
    Returns:
        Configured MixturePathGeneralizedKL loss
    """
    return MixturePathGeneralizedKL(
        time_scheduler=time_scheduler,
        reduction="mean",
        max_jump_coeff=10.0,  # Clip extreme jump coefficients
        loss_scale=loss_scale,  # Scale down the loss
        mask_token_id=mask_token_id  # Only calculate loss on masked positions
    )