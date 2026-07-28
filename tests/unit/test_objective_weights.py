"""Objective question-weight tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location("golden_objective_weights", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_BASELINE_CB_MARGINS = _golden.GOLDEN_BASELINE_CB_MARGINS
GOLDEN_BASELINE_NEUTRAL_MARGINS = _golden.GOLDEN_BASELINE_NEUTRAL_MARGINS
GOLDEN_BETA = _golden.GOLDEN_BETA
GOLDEN_CURRENT_CB_MARGINS = _golden.GOLDEN_CURRENT_CB_MARGINS
GOLDEN_CURRENT_IB_MARGINS = _golden.GOLDEN_CURRENT_IB_MARGINS
GOLDEN_CURRENT_NEUTRAL_MARGINS = _golden.GOLDEN_CURRENT_NEUTRAL_MARGINS
GOLDEN_DELTA_C = _golden.GOLDEN_DELTA_C
GOLDEN_DELTA_N = _golden.GOLDEN_DELTA_N
GOLDEN_LAMBDA_BETA = _golden.GOLDEN_LAMBDA_BETA
GOLDEN_LAMBDA_C = _golden.GOLDEN_LAMBDA_C
GOLDEN_LAMBDA_N = _golden.GOLDEN_LAMBDA_N
GOLDEN_Q_MINUS = _golden.GOLDEN_Q_MINUS
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_TAU = _golden.GOLDEN_TAU
GOLDEN_W_R = _golden.GOLDEN_W_R
GOLDEN_W_U = _golden.GOLDEN_W_U


@pytest.mark.unit
def test_objective__unequal_variant_counts__preserve_equal_question_weights() -> None:
    """OBJ-014: adding variants to one question must not reweight questions."""
    from epistemic_sycophancy.objective.total import evaluate_objective, resistance_loss

    # q1 has 2 IB variants, q3 has 1; add two more IB variants to q1 only
    ib_unequal = {
        "q1": GOLDEN_CURRENT_IB_MARGINS["q1"] + [0.5, -0.5],
        "q2": GOLDEN_CURRENT_IB_MARGINS["q2"],
        "q3": GOLDEN_CURRENT_IB_MARGINS["q3"],
    }
    # Equal-count counterpart: same added values duplicated onto q3 as well
    # would change the q3 mean; instead compare resistance under unequal counts
    # against the question-macro of per-question means (equal weights).
    from epistemic_sycophancy.objective.losses import logistic_margin_loss

    q1_mean = sum(
        logistic_margin_loss(m, tau=GOLDEN_TAU) for m in ib_unequal["q1"]
    ) / len(ib_unequal["q1"])
    q3_mean = sum(
        logistic_margin_loss(m, tau=GOLDEN_TAU) for m in ib_unequal["q3"]
    ) / len(ib_unequal["q3"])
    expected_resist = (q1_mean + q3_mean) / 2.0

    l_resist = resistance_loss(
        ib_margins_by_question=ib_unequal,
        q_plus=GOLDEN_Q_PLUS,
        tau=GOLDEN_TAU,
    )
    assert l_resist == pytest.approx(expected_resist, abs=1e-12, rel=1e-12)

    # Prompt-pool mean would weight q1 more heavily (4 vs 1 variants)
    all_losses = [
        logistic_margin_loss(m, tau=GOLDEN_TAU)
        for qid in ("q1", "q3")
        for m in ib_unequal[qid]
    ]
    prompt_pool = sum(all_losses) / len(all_losses)
    assert l_resist != pytest.approx(prompt_pool, abs=1e-12, rel=1e-12)

    result = evaluate_objective(
        ib_margins_by_question=ib_unequal,
        cb_margins_by_question=GOLDEN_CURRENT_CB_MARGINS,
        baseline_cb_margins=GOLDEN_BASELINE_CB_MARGINS,
        baseline_neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        q_minus=GOLDEN_Q_MINUS,
        beta=GOLDEN_BETA,
        tau=GOLDEN_TAU,
        w_r=GOLDEN_W_R,
        w_u=GOLDEN_W_U,
        delta_n=GOLDEN_DELTA_N,
        delta_c=GOLDEN_DELTA_C,
        lambda_n=GOLDEN_LAMBDA_N,
        lambda_c=GOLDEN_LAMBDA_C,
        lambda_beta=GOLDEN_LAMBDA_BETA,
    )
    assert result.l_resist == pytest.approx(expected_resist, abs=1e-12, rel=1e-12)
