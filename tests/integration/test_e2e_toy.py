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
    GOLDEN_KNOWN_BETA_L_BEHAVIOR,
    GOLDEN_KNOWN_BETA_L_BETA,
    GOLDEN_KNOWN_BETA_L_CORRECT,
    GOLDEN_KNOWN_BETA_L_NEUTRAL,
    GOLDEN_KNOWN_BETA_L_RECOVER,
    GOLDEN_KNOWN_BETA_L_RESIST,
    GOLDEN_KNOWN_BETA_L_TOTAL,
    GOLDEN_KNOWN_BETA_Q1N_DELTA,
    GOLDEN_KNOWN_BETA_Q1N_LATENTS,
    GOLDEN_KNOWN_BETA_Q1N_LATENTS_PRIME,
    GOLDEN_KNOWN_BETA_Q1N_LOGITS,
    GOLDEN_NEUTRAL_ACCURACY,
    GOLDEN_N_Q_MINUS,
    GOLDEN_N_Q_PLUS,
    GOLDEN_PRA_ALL,
    GOLDEN_PRA_MEAN,
    GOLDEN_Q_MINUS,
    GOLDEN_Q_PLUS,
    GOLDEN_SELECTIVITY,
    KNOWN_BETA,
    KNOWN_SCALES,
    KNOWN_SELECTED,
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


@pytest.mark.integration
def test_e2e_toy__zero_beta__matches_unhooked_pipeline() -> None:
    """E2E-002: β=0 hooked path matches unhooked baseline logits and margins."""
    from epistemic_sycophancy.evaluation.toy_e2e import run_toy_e2e_with_beta

    unhooked = run_toy_e2e_baseline(order_regime="CF")
    hooked = run_toy_e2e_with_beta(
        order_regime="CF",
        beta=(0.0, 0.0, 0.0),
        selected_indices=(0, 1, 2),
        scales=(1.0, 1.0, 1.0),
    )
    assert hooked.logits_by_prompt_id == unhooked.logits_by_prompt_id
    assert hooked.margins_by_prompt_id == unhooked.margins_by_prompt_id
    assert hooked.neutral_margins == unhooked.neutral_margins
    assert hooked.ib_margins == unhooked.ib_margins
    assert hooked.cb_margins == unhooked.cb_margins


@pytest.mark.integration
def test_e2e_toy__known_beta__matches_hand_computed_latents_delta_logits_and_objective() -> (
    None
):
    """E2E-003: known β matches hand-computed latents, Δx, logits, and objective."""
    from epistemic_sycophancy.evaluation.toy_e2e import (
        evaluate_toy_e2e_objective,
        inspect_toy_e2e_prompt,
    )

    detail = inspect_toy_e2e_prompt(
        prompt_id="CF:q1:N:0",
        beta=KNOWN_BETA,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
    )
    assert detail.latents == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_LATENTS, abs=1e-12, rel=1e-12
    )
    assert detail.latents_prime == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_LATENTS_PRIME, abs=1e-12, rel=1e-12
    )
    assert detail.residual_delta == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_DELTA, abs=1e-12, rel=1e-12
    )
    assert detail.logits == pytest.approx(
        GOLDEN_KNOWN_BETA_Q1N_LOGITS, abs=1e-12, rel=1e-12
    )

    objective = evaluate_toy_e2e_objective(
        order_regime="CF",
        beta=KNOWN_BETA,
        selected_indices=KNOWN_SELECTED,
        scales=KNOWN_SCALES,
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.1,
        delta_c=0.1,
        lambda_n=0.1,
        lambda_c=0.1,
        lambda_beta=0.1,
    )
    assert objective.l_resist == pytest.approx(
        GOLDEN_KNOWN_BETA_L_RESIST, abs=1e-12, rel=1e-12
    )
    assert objective.l_recover == pytest.approx(
        GOLDEN_KNOWN_BETA_L_RECOVER, abs=1e-12, rel=1e-12
    )
    assert objective.l_behavior == pytest.approx(
        GOLDEN_KNOWN_BETA_L_BEHAVIOR, abs=1e-12, rel=1e-12
    )
    assert objective.l_neutral == pytest.approx(
        GOLDEN_KNOWN_BETA_L_NEUTRAL, abs=1e-12, rel=1e-12
    )
    assert objective.l_correct == pytest.approx(
        GOLDEN_KNOWN_BETA_L_CORRECT, abs=1e-12, rel=1e-12
    )
    assert objective.l_beta == pytest.approx(
        GOLDEN_KNOWN_BETA_L_BETA, abs=1e-12, rel=1e-12
    )
    assert objective.l_total == pytest.approx(
        GOLDEN_KNOWN_BETA_L_TOTAL, abs=1e-12, rel=1e-12
    )
