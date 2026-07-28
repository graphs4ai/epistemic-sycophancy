"""Projected Adam over β only (Phase H; DEC-031)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Adam

from epistemic_sycophancy.objective.total import accumulate_objective_batches


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


def _partition_questions(
    question_ids: Sequence[object],
    *,
    microbatch_questions: int,
) -> list[list[object]]:
    """Partition questions into contiguous microbatches of the given size."""
    batches: list[list[object]] = []
    batch: list[object] = []
    for question_id in question_ids:
        batch.append(question_id)
        if len(batch) >= microbatch_questions:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def microbatch_objective_gradient(
    optimizer: ProjectedAdam,
    *,
    beta: object,
    question_ids: Sequence[object],
    ib_margin_const: Mapping[object, Sequence[float]],
    ib_margin_jac: Mapping[object, Sequence[object]],
    cb_margin_const: Mapping[object, Sequence[float]],
    cb_margin_jac: Mapping[object, Sequence[object]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    neutral_margin_const: Mapping[object, float],
    neutral_margin_jac: Mapping[object, object],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
) -> tuple[float, list[float]]:
    """Full-split ∇β via question microbatches (OPT-007 / DEC-027).

    Numerators and denominators are accumulated exactly — never mean-of-batch-means.
    """
    batches = _partition_questions(
        question_ids,
        microbatch_questions=optimizer.adam_microbatch_questions,
    )
    return accumulate_objective_batches(
        beta=beta,
        question_batches=batches,
        ib_margin_const=ib_margin_const,
        ib_margin_jac=ib_margin_jac,
        cb_margin_const=cb_margin_const,
        cb_margin_jac=cb_margin_jac,
        baseline_cb_margins=baseline_cb_margins,
        baseline_neutral_margins=baseline_neutral_margins,
        neutral_margin_const=neutral_margin_const,
        neutral_margin_jac=neutral_margin_jac,
        q_plus=q_plus,
        q_minus=q_minus,
        tau=tau,
        w_r=w_r,
        w_u=w_u,
        delta_n=delta_n,
        delta_c=delta_c,
        lambda_n=lambda_n,
        lambda_c=lambda_c,
        lambda_beta=lambda_beta,
    )
