# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MixturePathGeneralizedKL(_Loss):
    r"""A generalized KL loss for discrete flow matching.
    A class that measures the generalized KL of a discrete flow model p_{1|t} w.r.t. 
    a probability path. Note: this class is assuming that the model is trained on the same path.

    For a model trained on a space S = T^d, T = [K] = {1,2,...,K}, the loss is given by

    l_i(x_1, x_t, t) = -(dot_kappa_t)/(1-kappa_t) * [p_{1|t}(x_t^i|x_t) - delta_{x^i_1}(x_t^i) + 
                        (1-delta_{x^i_1}(x_t^i)) * log(p_{1|t}(x_1^i|x_t))]

    where kappa_t is the scheduler parameter.

    Args:
        alpha_t (Tensor): scheduler parameter alpha at time t
        d_alpha_t (Tensor): derivative of alpha_t with respect to time
        reduction (str, optional): Specify the reduction to apply to the output 
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(None, None, reduction)

    def forward(self, logits: Tensor, x_1: Tensor, x_t: Tensor, t: Tensor, 
                alpha_t: Tensor, d_alpha_t: Tensor) -> Tensor:
        r"""Evaluates the generalized KL loss.

        Args:
            logits (Tensor): posterior model output (i.e., softmax(logits) = p_{1|t}(x|x_t)), 
                           shape (batch, seq_len, vocab_size).
            x_1 (Tensor): target data point x_1, shape (batch, seq_len).
            x_t (Tensor): conditional sample at x_t ~ p_t(·|x_1), shape (batch, seq_len).
            t (Tensor): times in [0,1], shape (batch,).
            alpha_t (Tensor): scheduler parameter alpha at time t, shape (batch,).
            d_alpha_t (Tensor): derivative of alpha_t, shape (batch,).

        Raises:
            ValueError: reduction value must be one of 'none' | 'mean' | 'sum'.

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

        # Calculate jump coefficient
        jump_coefficient = (d_alpha_t / (1 - alpha_t))[(...,) + (None,) * (x_1.dim() - 1)]
        jump_coefficient = jump_coefficient.repeat(1, *x_1_shape[1:])
        
        # Delta function: 1 if x_t == x_1, 0 otherwise
        delta_x1_xt = (x_t == x_1).to(log_p_1t.dtype)

        # Compute the generalized KL loss
        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")