"""Objective invariance property tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location(
    "golden_objective_invariance", _GOLDEN_PATH
)
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


def _evaluate(ib, cb, baseline_cb, baseline_n, current_n):
    from epistemic_sycophancy.objective.total import evaluate_objective

    return evaluate_objective(
        ib_margins_by_question=ib,
        cb_margins_by_question=cb,
        baseline_cb_margins=baseline_cb,
        baseline_neutral_margins=baseline_n,
        current_neutral_margins=current_n,
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


def _permute_list(values: list[float], seed: int) -> list[float]:
    order = list(range(len(values)))
    # Deterministic Fisher–Yates with fixed seed
    state = seed % (2**31)
    for i in range(len(order) - 1, 0, -1):
        state = (1103515245 * state + 12345) % (2**31)
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    return [values[i] for i in order]


def _qid_salt(qid: object) -> int:
    return sum(ord(c) for c in str(qid))


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=50, deadline=None)
def test_objective__permuting_prompt_rows__does_not_change_result(seed: int) -> None:
    """OBJ-012: permuting variant rows within questions leaves L_total unchanged."""
    ib = {
        qid: _permute_list(list(margins), seed + _qid_salt(qid))
        for qid, margins in GOLDEN_CURRENT_IB_MARGINS.items()
    }
    cb: dict[str, list[float]] = {}
    baseline_cb: dict[str, list[float]] = {}
    for qid, margins in GOLDEN_CURRENT_CB_MARGINS.items():
        order_seed = seed + _qid_salt(qid) + 17
        cb[qid] = _permute_list(list(margins), order_seed)
        if qid in GOLDEN_BASELINE_CB_MARGINS:
            baseline_cb[qid] = _permute_list(
                list(GOLDEN_BASELINE_CB_MARGINS[qid]), order_seed
            )

    baseline = _evaluate(
        GOLDEN_CURRENT_IB_MARGINS,
        GOLDEN_CURRENT_CB_MARGINS,
        GOLDEN_BASELINE_CB_MARGINS,
        GOLDEN_BASELINE_NEUTRAL_MARGINS,
        GOLDEN_CURRENT_NEUTRAL_MARGINS,
    )
    permuted = _evaluate(
        ib,
        cb,
        baseline_cb,
        GOLDEN_BASELINE_NEUTRAL_MARGINS,
        GOLDEN_CURRENT_NEUTRAL_MARGINS,
    )
    assert permuted.l_total == pytest.approx(baseline.l_total, abs=1e-12, rel=1e-12)
