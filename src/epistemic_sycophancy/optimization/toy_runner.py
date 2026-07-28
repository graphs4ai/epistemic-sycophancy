"""Thin projected-Adam runner for affine-margin toy objectives (OPT-GATE-001)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from epistemic_sycophancy.objective.total import evaluate_objective_with_grad
from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam


@dataclass(frozen=True)
class ToyAdamRunResult:
    """Outcome of a pinned toy projected-Adam run."""

    l_initial: float
    l_final: float
    beta_final: list[float]
    beta_trajectory: tuple[tuple[float, ...], ...]


def run_projected_adam_affine(
    *,
    beta0: Sequence[float],
    n_steps: int,
    adam_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    adam_microbatch_questions: int,
    beta_lower: float,
    beta_upper: float,
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
) -> ToyAdamRunResult:
    """Run projected Adam on an affine-margin objective; return initial/final loss."""
    del question_ids  # full-split path uses baseline_neutral keys
    del adam_microbatch_questions  # gate uses full-split grad each step
    beta = torch.tensor(list(beta0), dtype=torch.float64, requires_grad=True)
    optimizer = ProjectedAdam(
        beta=beta,
        adam_lr=adam_lr,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        adam_microbatch_questions=1,
        beta_lower=beta_lower,
        beta_upper=beta_upper,
    )
    kwargs = dict(
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
    l_initial, grad0 = evaluate_objective_with_grad(beta=beta.detach(), **kwargs)
    trajectory: list[tuple[float, ...]] = [tuple(beta.detach().tolist())]
    # First step uses grad0; subsequent steps recompute
    for step in range(int(n_steps)):
        optimizer.zero_grad(set_to_none=True)
        if step == 0:
            grad = grad0
            loss = l_initial
        else:
            loss, grad = evaluate_objective_with_grad(beta=beta.detach(), **kwargs)
        beta.grad = torch.tensor(grad, dtype=torch.float64)
        optimizer.step()
        trajectory.append(tuple(beta.detach().tolist()))
        del loss

    l_final, _ = evaluate_objective_with_grad(beta=beta.detach(), **kwargs)
    return ToyAdamRunResult(
        l_initial=float(l_initial),
        l_final=float(l_final),
        beta_final=beta.detach().tolist(),
        beta_trajectory=tuple(trajectory),
    )
