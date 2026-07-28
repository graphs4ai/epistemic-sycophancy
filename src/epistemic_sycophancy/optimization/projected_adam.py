"""Projected Adam over β only (Phase H; DEC-031)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Adam


class ProjectedAdam:
    """Adam on a single β parameter with post-step box projection."""

    def __init__(
        self,
        *,
        beta: Tensor,
        adam_lr: float,
        adam_beta1: float,
        adam_beta2: float,
        adam_eps: float,
        adam_microbatch_questions: int,
        beta_lower: float,
        beta_upper: float,
    ) -> None:
        if adam_lr is None or adam_beta1 is None or adam_beta2 is None or adam_eps is None:
            raise ValueError("Adam hyperparameters are required (DEC-031)")
        if adam_microbatch_questions is None or int(adam_microbatch_questions) < 1:
            raise ValueError("adam_microbatch_questions must be a positive int (DEC-031)")
        if not (beta_lower <= beta_upper <= 0):
            raise ValueError(
                "suppression-only bounds require beta_lower <= beta_upper <= 0"
            )
        if not isinstance(beta, Tensor):
            raise TypeError("beta must be a torch.Tensor Parameter/leaf")
        self.beta = beta
        self.adam_lr = float(adam_lr)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self.adam_microbatch_questions = int(adam_microbatch_questions)
        self.beta_lower = float(beta_lower)
        self.beta_upper = float(beta_upper)
        self._optimizer = Adam(
            [self.beta],
            lr=self.adam_lr,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

    @property
    def torch_optimizer(self) -> Adam:
        """Underlying torch Adam (β-only param group)."""
        return self._optimizer

    def step(self) -> None:
        """Take one Adam step then clamp β to configured bounds."""
        self._optimizer.step()
        with torch.no_grad():
            self.beta.clamp_(min=self.beta_lower, max=self.beta_upper)

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear β gradients."""
        self._optimizer.zero_grad(set_to_none=set_to_none)
