# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
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


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from :math:`\kappa_t`.

        Args:
            kappa (Tensor): :math:`\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        r"""
        Computes :math:`t` from the signal-to-noise ratio :math:`\frac{\alpha_t}{\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class PolynomialConvexScheduler(ConvexScheduler):
    """Polynomial Scheduler adapted for discrete diffusion training."""

    def __init__(self, n: Union[float, int] = 1.0) -> None:
        assert isinstance(n, (float, int)), f"`n` must be a float or int. Got {type(n)=}."
        assert n > 0, f"`n` must be positive. Got {n=}."
        self.n = n

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t**self.n,
            sigma_t=1 - t**self.n,
            d_alpha_t=self.n * (t ** (self.n - 1)),
            d_sigma_t=-self.n * (t ** (self.n - 1)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return torch.pow(kappa, 1.0 / self.n)


class LinearScheduler(ConvexScheduler):
    """Linear scheduler for discrete diffusion training."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa


@dataclass
class DiscretePathSample:
    """Represents a sample from a discrete probability path."""
    x_t: Tensor = field(metadata={"help": "Sample at time t"})
    x_1: Tensor = field(metadata={"help": "Target data point"})
    x_0: Tensor = field(metadata={"help": "Source data point"})
    t: Tensor = field(metadata={"help": "Time"})


def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """Expand input_tensor to match the shape of expand_to."""
    while input_tensor.dim() < expand_to.dim():
        input_tensor = input_tensor.unsqueeze(-1)
    return input_tensor.expand_as(expand_to)


def unsqueeze_to_match(source: Tensor, target: Tensor) -> Tensor:
    """Unsqueeze source tensor to match target tensor dimensions."""
    while source.dim() < target.dim():
        source = source.unsqueeze(-1)
    return source


class MixtureDiscreteProbPath:
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, 
    determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\\sigma_t`.

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\\sigma_t`.
    """

    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(scheduler, ConvexScheduler), "Scheduler must be a ConvexScheduler."
        self.scheduler = scheduler

    def assert_sample_shape(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        r"""Assert that input tensors have compatible shapes."""
        assert x_0.shape == x_1.shape, f"x_0 and x_1 must have same shape, got {x_0.shape} vs {x_1.shape}"
        assert t.shape[0] == x_0.shape[0], f"Batch dimension mismatch: t={t.shape[0]}, x_0={x_0.shape[0]}"

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        r"""Sample from the discrete probability path.
        
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t
        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(self, posterior_logits: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert the factorized posterior to velocity.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1)
        vocabulary_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)


class MixturePathGeneralizedKL(_Loss):
    r"""A generalized KL loss for discrete flow matching adapted for diffusion training.
    
    This class measures the generalized KL of a discrete flow model w.r.t. a probability 
    path given by ``path``. It's adapted to work with the discrete diffusion training setup.

    Args:
        path (MixtureDiscreteProbPath): Probability path (x-prediction training).
        reduction (str, optional): Specify the reduction to apply to the output 
            ``'none'`` | ``'mean'`` | ``'sum'``. Defaults to 'mean'.
    """

    def __init__(self, path: MixtureDiscreteProbPath, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)
        self.path = path

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor, 
                attention_mask: Tensor = None) -> Tensor:
        r"""Evaluates the generalized KL loss adapted for diffusion training.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(``logits``) = p_{1|t}(x|x_t)), 
                           shape (batch, seq_len, vocab_size).
            x_1 (Tensor): target data point (clean tokens), shape (batch, seq_len).
            x_t (Tensor): conditional sample at x_t ~ p_t(·|x_1) (noised tokens), 
                         shape (batch, seq_len).
            t (Tensor): times in [0,1], shape (batch).
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

        scheduler_output = self.path.scheduler(t)

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


def create_time_scheduler(num_timesteps: int, scheduler_type: str = "linear", device: str = "cpu") -> Tensor:
    """
    Create a time scheduler tensor compatible with your training loop.
    
    Args:
        num_timesteps (int): Number of timesteps
        scheduler_type (str): Type of scheduler ("linear" or "polynomial")
        device (str): Device to place the tensor on
        
    Returns:
        Tensor: Time scheduler tensor of shape (num_timesteps,)
    """
    if scheduler_type == "linear":
        return torch.linspace(1 / num_timesteps, 1, steps=num_timesteps, dtype=torch.float32, device=device)
    elif scheduler_type == "polynomial":
        # Polynomial schedule with n=2 for more aggressive noise schedule
        t = torch.linspace(0, 1, steps=num_timesteps, dtype=torch.float32, device=device)
        return t ** 2
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")