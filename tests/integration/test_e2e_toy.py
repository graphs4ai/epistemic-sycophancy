"""Toy end-to-end integration tests (Phase J E2E / spec §21)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_baseline
from tests.fixtures.e2e.corpus import (
    GOLDEN_CF_BASELINE_LOGITS,
    GOLDEN_CF_BASELINE_MARGINS,
    GOLDEN_CF_CB_MARGINS,
    GOLDEN_CF_IB_MARGINS,
    GOLDEN_CF_NEUTRAL_MARGINS,
    GOLDEN_CBR,
    GOLDEN_FTW,
    GOLDEN_NEUTRAL_ACCURACY,
    GOLDEN_N_Q_MINUS,
    GOLDEN_N_Q_PLUS,
    GOLDEN_PRA_ALL,
    GOLDEN_PRA_MEAN,
    GOLDEN_Q_MINUS,
    GOLDEN_Q_PLUS,
    GOLDEN_SELECTIVITY,
)


@pytest.mark.integration
def test_e2e_toy__baseline__matches_hand_computed_logits_margins_partitions_and_metrics() -> (
    None
):
    """E2E-001: unintervened toy pipeline matches hand-computed CF goldens."""
    result = run_toy_e2e_baseline(order_regime="CF")

    for prompt_id, expected_logits in GOLDEN_CF_BASELINE_LOGITS.items():
        got = result.logits_by_prompt_id[prompt_id]
        assert got == pytest.approx(expected_logits, abs=1e-12, rel=1e-12)

    for prompt_id, expected_margin in GOLDEN_CF_BASELINE_MARGINS.items():
        assert result.margins_by_prompt_id[prompt_id] == pytest.approx(
            expected_margin, abs=1e-12, rel=1e-12
        )

    assert result.neutral_margins == pytest.approx(
        GOLDEN_CF_NEUTRAL_MARGINS, abs=1e-12, rel=1e-12
    )
    for qid, expected in GOLDEN_CF_IB_MARGINS.items():
        assert result.ib_margins[qid] == pytest.approx(expected, abs=1e-12, rel=1e-12)
    for qid, expected in GOLDEN_CF_CB_MARGINS.items():
        assert result.cb_margins[qid] == pytest.approx(expected, abs=1e-12, rel=1e-12)

    assert result.partition.q_plus == GOLDEN_Q_PLUS
    assert result.partition.q_minus == GOLDEN_Q_MINUS
    assert result.metrics.neutral_accuracy == pytest.approx(
        GOLDEN_NEUTRAL_ACCURACY, abs=1e-12, rel=1e-12
    )
    assert result.metrics.ftw == pytest.approx(GOLDEN_FTW, abs=1e-12, rel=1e-12)
    assert result.metrics.cbr == pytest.approx(GOLDEN_CBR, abs=1e-12, rel=1e-12)
    assert result.metrics.selectivity == pytest.approx(
        GOLDEN_SELECTIVITY, abs=1e-12, rel=1e-12
    )
    assert result.metrics.pra_mean == pytest.approx(
        GOLDEN_PRA_MEAN, abs=1e-12, rel=1e-12
    )
    assert result.metrics.pra_all == pytest.approx(
        GOLDEN_PRA_ALL, abs=1e-12, rel=1e-12
    )
    assert result.metrics.n_q_plus == GOLDEN_N_Q_PLUS
    assert result.metrics.n_q_minus == GOLDEN_N_Q_MINUS
