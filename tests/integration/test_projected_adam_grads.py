"""Projected Adam microbatch gradient tests (Phase H OPT-007)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
def test_projected_adam__microbatch_gradient__matches_unbatched_full_objective_gradient() -> None:
    """OPT-007: microbatch ∇β matches unbatched full-objective ∇β (DEC-027)."""
    from epistemic_sycophancy.objective.total import evaluate_objective_with_grad
    from epistemic_sycophancy.optimization.projected_adam import (
        ProjectedAdam,
        microbatch_objective_gradient,
    )

    beta = torch.tensor([-0.5, -0.25], dtype=torch.float64)
    q_plus = frozenset({"q1", "q3"})
    q_minus = frozenset({"q2"})

    kwargs = dict(
        ib_margin_const={
            "q1": [1.0, -0.5],
            "q3": [0.2],
        },
        ib_margin_jac={
            "q1": [
                torch.tensor([0.5, -0.2], dtype=torch.float64),
                torch.tensor([-0.1, 0.3], dtype=torch.float64),
            ],
            "q3": [torch.tensor([0.1, 0.0], dtype=torch.float64)],
        },
        cb_margin_const={
            "q1": [2.2],
            "q2": [2.0, -1.0],
            "q3": [1.05],
        },
        cb_margin_jac={
            "q1": [torch.tensor([0.1, 0.1], dtype=torch.float64)],
            "q2": [
                torch.tensor([0.3, -0.1], dtype=torch.float64),
                torch.tensor([-0.4, 0.2], dtype=torch.float64),
            ],
            "q3": [torch.tensor([0.0, -0.2], dtype=torch.float64)],
        },
        baseline_cb_margins={"q1": [2.5], "q3": [1.0]},
        baseline_neutral_margins={"q1": 2.0, "q2": -1.0, "q3": 0.5},
        neutral_margin_const={"q1": 1.4, "q2": -0.2, "q3": 0.8},
        neutral_margin_jac={
            "q1": torch.tensor([0.2, 0.0], dtype=torch.float64),
            "q2": torch.tensor([-0.1, 0.1], dtype=torch.float64),
            "q3": torch.tensor([0.0, 0.05], dtype=torch.float64),
        },
        q_plus=q_plus,
        q_minus=q_minus,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.25,
        delta_c=0.10,
        lambda_n=2.0,
        lambda_c=1.5,
        lambda_beta=0.1,
    )

    full_loss, full_grad = evaluate_objective_with_grad(beta=beta, **kwargs)

    optimizer = ProjectedAdam(
        beta=beta.clone().requires_grad_(True),
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=0.0,
    )
    question_ids = list(kwargs["baseline_neutral_margins"].keys())
    micro_loss, micro_grad = microbatch_objective_gradient(
        optimizer,
        beta=beta,
        question_ids=question_ids,
        **kwargs,
    )

    assert micro_loss == pytest.approx(full_loss, abs=1e-8, rel=1e-6)
    assert list(micro_grad) == pytest.approx(list(full_grad), abs=1e-8, rel=1e-6)
