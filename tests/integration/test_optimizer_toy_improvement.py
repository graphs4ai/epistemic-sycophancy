"""Phase H optimizer gate: toy projected Adam improvement (OPT-GATE-001)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "optimization"
    / "toy_gate_objective.py"
)
_spec = importlib.util.spec_from_file_location("toy_gate_objective", _FIXTURE)
assert _spec is not None and _spec.loader is not None
_fix = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _fix
_spec.loader.exec_module(_fix)


@pytest.mark.integration
def test_optimizer_gate__projected_adam__improves_pinned_toy_objective_within_bounds() -> None:
    """OPT-GATE-001: projected Adam reduces pinned toy L_total within CFG-004 bounds."""
    from epistemic_sycophancy.optimization.toy_runner import run_projected_adam_affine

    result = run_projected_adam_affine(
        beta0=_fix.GATE_BETA0,
        n_steps=_fix.GATE_N_STEPS,
        adam_lr=_fix.GATE_ADAM_LR,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=_fix.GATE_BETA_LOWER,
        beta_upper=_fix.GATE_BETA_UPPER,
        question_ids=_fix.GATE_QUESTION_IDS,
        ib_margin_const=_fix.GATE_IB_MARGIN_CONST,
        ib_margin_jac=_fix.GATE_IB_MARGIN_JAC,
        cb_margin_const=_fix.GATE_CB_MARGIN_CONST,
        cb_margin_jac=_fix.GATE_CB_MARGIN_JAC,
        baseline_cb_margins=_fix.GATE_BASELINE_CB,
        baseline_neutral_margins=_fix.GATE_BASELINE_NEUTRAL,
        neutral_margin_const=_fix.GATE_NEUTRAL_CONST,
        neutral_margin_jac=_fix.GATE_NEUTRAL_JAC,
        q_plus=_fix.GATE_Q_PLUS,
        q_minus=_fix.GATE_Q_MINUS,
        tau=_fix.GATE_TAU,
        w_r=_fix.GATE_W_R,
        w_u=_fix.GATE_W_U,
        delta_n=_fix.GATE_DELTA_N,
        delta_c=_fix.GATE_DELTA_C,
        lambda_n=_fix.GATE_LAMBDA_N,
        lambda_c=_fix.GATE_LAMBDA_C,
        lambda_beta=_fix.GATE_LAMBDA_BETA,
    )

    assert result.l_final < result.l_initial
    assert all(
        _fix.GATE_BETA_LOWER <= v <= _fix.GATE_BETA_UPPER for v in result.beta_final
    )
    assert all(
        _fix.GATE_BETA_LOWER <= v <= _fix.GATE_BETA_UPPER
        for beta in result.beta_trajectory
        for v in beta
    )
