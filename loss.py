# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Optional


class SchedulerOutput:
    """Simple container for scheduler outputs to match the expected interface"""
    def __init__(self, alpha_t: Tensor, d_alpha_t: Tensor):
        self.alpha_t = alpha_t
        self.d_alpha_t = d_alpha_t


class MixtureDiscreteProbPath:
    """Simple probability path implementation for discrete flow matching"""
    def __init__(self, scheduler_fn=None):
        self.scheduler_fn = scheduler_fn
    
    def scheduler(self, t: Tensor) -> SchedulerOutput:
        if self.scheduler_fn is not None:
            return self.scheduler_fn(t)
        
        # Default linear scheduler
        alpha_t = 1.0 - t
        d_alpha_t = -torch.ones_like(t)
        return SchedulerOutput(alpha_t, d_alpha_t)


class MixturePathGeneralizedKL(_Loss):
    r"""A generalized KL loss for discrete flow matching.
    A class that measures the generalized KL of a discrete flow model :math:`p_{1|t}` w.r.t. a probability path given by ``path``. Note: this class is assuming that the model is trained on the same path.

    For a model trained on a space :math:`\mathcal{S} = \mathcal{T}^d`, :math:`\mathcal{T} = [K] = \set{1,2,\ldots,K}`, the loss is given by

    .. math::
            \ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1-\kappa_t} \biggr[  p_{1|t}(x_t^i|x_t) -\delta_{x^i_1}(x_t^i) + (1-\delta_{x^i_1}(x_t^i))\left(\log p_{1|t}(x_1^i|x_t)\right)\biggr],

    where :math:`\kappa_t` is the scheduler associated with ``path``.

    Args:
        path (MixtureDiscreteProbPath, optional): Probability path (x-prediction training). If None, uses default linear path.
        reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """

    def __init__(self, path: Optional[MixtureDiscreteProbPath] = None, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = path if path is not None else MixtureDiscreteProbPath()

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Evaluates the generalized KL loss.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(``logits``) :math:`=p_{1|t}(x|x_t)`), shape (batch, seq_len, vocab_size).
            x_1 (Tensor): target data point :math:`x_1 \sim q`, shape (batch, seq_len).
            x_t (Tensor): conditional sample at :math:`x_t \sim p_t(\cdot|x_1)`, shape (batch, seq_len).
            t (Tensor): times in :math:`[0,1]`, shape (batch,).

        Raises:
            ValueError: reduction value must be one of ``'none'`` | ``'mean'`` | ``'sum'``.

        Returns:
            Tensor: Generalized KL loss.
        """
        batch_size, seq_len = x_1.shape
        vocab_size = logits.shape[-1]
        
        # Flatten for easier processing
        logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
        x_1_flat = x_1.view(-1)  # (batch * seq_len,)
        x_t_flat = x_t.view(-1)  # (batch * seq_len,)
        
        # Create mask for valid positions (not -100)
        valid_mask = (x_1_flat != -100)
        
        if not valid_mask.any():
            # If no valid positions, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Apply mask to get only valid positions
        logits_valid = logits_flat[valid_mask]  # (num_valid, vocab_size)
        x_1_valid = x_1_flat[valid_mask]  # (num_valid,)
        x_t_valid = x_t_flat[valid_mask]  # (num_valid,)
        
        # Extract x_1 value of log(p_{1|t}(x|x_t))
        log_p_1t = torch.log_softmax(logits_valid, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1_valid.unsqueeze(-1)).squeeze(-1)
        
        # Extract x_t value of p_{1|t}(x|x_t)
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t_valid.unsqueeze(-1)).squeeze(-1)
        
        # Get scheduler output
        scheduler_output = self.path.scheduler(t)
        
        # Expand jump coefficient to match the number of valid positions
        # We need to map back from valid positions to their original batch indices
        batch_indices = torch.arange(batch_size, device=logits.device).repeat_interleave(seq_len)[valid_mask]
        jump_coefficient = scheduler_output.d_alpha_t[batch_indices] / (1 - scheduler_output.alpha_t[batch_indices])
        
        # Calculate delta function: 1 if x_t == x_1, 0 otherwise
        delta_x1_xt = (x_t_valid == x_1_valid).float()
        
        # Calculate the generalized KL loss
        loss_per_position = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )
        
        if self.reduction == "mean":
            return torch.mean(loss_per_position)
        elif self.reduction == "sum":
            return torch.sum(loss_per_position)
        elif self.reduction == "none":
            # Reshape back to original shape, filling invalid positions with 0
            loss_full = torch.zeros_like(x_1_flat, dtype=loss_per_position.dtype, device=loss_per_position.device)
            loss_full[valid_mask] = loss_per_position
            return loss_full.view(batch_size, seq_len)
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")


class SimplifiedFlowLoss(_Loss):
    """Simplified version of the flow loss that's easier to integrate with existing training code"""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(None, None, reduction)
    
    def forward(self, logits: Tensor, target: Tensor, noised_input: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            target: Original clean tokens (batch, seq_len) - may contain -100 for ignored positions
            noised_input: Noised input tokens (batch, seq_len)
            t: Time values (batch,)
        """
        # Create a simple flow-based loss
        # For positions where target != -100, compute cross-entropy loss weighted by time
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten tensors
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target.view(-1)
        noised_flat = noised_input.view(-1)
        
        # Create mask for valid positions
        valid_mask = (target_flat != -100)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Apply mask
        logits_valid = logits_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        noised_valid = noised_flat[valid_mask]
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits_valid, dim=-1)
        
        # Get log probability of target tokens
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_valid.unsqueeze(-1)).squeeze(-1)
        
        # Time-weighted loss - higher weight for later times (more denoising needed)
        batch_indices = torch.arange(batch_size, device=logits.device).repeat_interleave(seq_len)[valid_mask]
        time_weights = t[batch_indices]
        
        # Basic cross-entropy loss with time weighting
        loss_per_position = -target_log_probs * time_weights
        
        if self.reduction == "mean":
            return torch.mean(loss_per_position)
        elif self.reduction == "sum":
            return torch.sum(loss_per_position)
        elif self.reduction == "none":
            loss_full = torch.zeros_like(target_flat, dtype=loss_per_position.dtype, device=loss_per_position.device)
            loss_full[valid_mask] = loss_per_position
            return loss_full.view(batch_size, seq_len)
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")