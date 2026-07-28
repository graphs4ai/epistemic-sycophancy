"""Batched vs unbatched objective equivalence (Phase G OBJ-015 / DEC-027)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.integration
def test_objective__batch_partitioning__does_not_change_full_split_loss_or_gradient() -> None:
    """OBJ-015: question batches must match full-split L and ∂L/∂β (DEC-027)."""
    from epistemic_sycophancy.objective.total import (
        accumulate_objective_batches,
        evaluate_objective_with_grad,
    )

    beta = torch.tensor([-0.5, -0.25], dtype=torch.float64)
    q_plus = frozenset({"q1", "q3"})
    q_minus = frozenset({"q2"})

    ib_margin_const = {
        "q1": [1.0, -0.5],
        "q3": [0.2],
    }
    ib_margin_jac = {
        "q1": [
            torch.tensor([0.5, -0.2], dtype=torch.float64),
            torch.tensor([-0.1, 0.3], dtype=torch.float64),
        ],
        "q3": [torch.tensor([0.1, 0.0], dtype=torch.float64)],
    }
    cb_margin_const = {
        "q1": [2.2],
        "q2": [2.0, -1.0],
        "q3": [1.05],
    }
    cb_margin_jac = {
        "q1": [torch.tensor([0.1, 0.1], dtype=torch.float64)],
        "q2": [
            torch.tensor([0.3, -0.1], dtype=torch.float64),
            torch.tensor([-0.4, 0.2], dtype=torch.float64),
        ],
        "q3": [torch.tensor([0.0, -0.2], dtype=torch.float64)],
    }
    baseline_cb_margins = {
        "q1": [2.5],
        "q3": [1.0],
    }
    baseline_neutral_margins = {"q1": 2.0, "q2": -1.0, "q3": 0.5}
    neutral_margin_const = {"q1": 1.4, "q2": -0.2, "q3": 0.8}
    neutral_margin_jac = {
        "q1": torch.tensor([0.2, 0.0], dtype=torch.float64),
        "q2": torch.tensor([-0.1, 0.1], dtype=torch.float64),
        "q3": torch.tensor([0.0, 0.05], dtype=torch.float64),
    }

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

    # Uneven question batches — mean-of-batch-means would bias
    batches = [["q1"], ["q2", "q3"]]
    batch_loss, batch_grad = accumulate_objective_batches(
        beta=beta,
        question_batches=batches,
        **kwargs,
    )

    assert batch_loss == pytest.approx(full_loss, abs=1e-8, rel=1e-6)
    assert list(batch_grad) == pytest.approx(list(full_grad), abs=1e-8, rel=1e-6)
    assert len(full_grad) == int(beta.numel())
