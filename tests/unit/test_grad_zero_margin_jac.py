"""GRAD-001: zero margin Jacobian at β=0 yields ~0 ∂L/∂β (characterization)."""

from __future__ import annotations

import math

import pytest
import torch


@pytest.mark.unit
def test_grad__zero_margin_jac_at_beta_zero__grad_approximately_zero() -> None:
    """GRAD-001: with G≡0, δ_n=δ_c=0, β=0, only λ_β·mean(|β|) remains → grad ≈ 0.

    Pins the mathematical consequence of all-zero margin Jacobians. Does not
    endorse production defaulting to G=0 (see DEC-084 / GRAD-002+).
    """
    from epistemic_sycophancy.objective.total import evaluate_objective_with_grad

    m = 2
    zero_row = torch.zeros(m, dtype=torch.float64)
    beta = torch.zeros(m, dtype=torch.float64)
    q_plus = frozenset({"q1"})
    q_minus = frozenset({"q2"})

    loss, grad = evaluate_objective_with_grad(
        beta=beta,
        ib_margin_const={"q1": (0.25,), "q2": (0.25,)},
        ib_margin_jac={"q1": [zero_row.clone()], "q2": [zero_row.clone()]},
        cb_margin_const={"q1": (0.75,), "q2": (0.75,)},
        cb_margin_jac={"q1": [zero_row.clone()], "q2": [zero_row.clone()]},
        baseline_cb_margins={"q1": (0.75,), "q2": (0.75,)},
        baseline_neutral_margins={"q1": 1.0, "q2": -0.5},
        neutral_margin_const={"q1": 1.0, "q2": -0.5},
        neutral_margin_jac={"q1": zero_row.clone(), "q2": zero_row.clone()},
        q_plus=q_plus,
        q_minus=q_minus,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.0,
        delta_c=0.0,
        lambda_n=0.0,
        lambda_c=0.0,
        lambda_beta=0.01,
    )
    del loss
    assert len(grad) == m
    grad_norm = math.sqrt(sum(float(g) * float(g) for g in grad))
    assert grad_norm == pytest.approx(0.0, abs=1e-12)
