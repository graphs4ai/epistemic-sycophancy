"""Property tests for behavioral metric bounds (METRIC-009)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.metrics.baseline_partition import (
    build_baseline_partition,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import compute_behavioral_metrics

_finite = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=-10.0,
    max_value=10.0,
)


@pytest.mark.property
@given(
    m_n1=_finite,
    m_n2=_finite,
    m_n3=_finite,
    ib1=_finite,
    ib2=_finite,
    ib3=_finite,
    cb1=_finite,
    cb2=_finite,
)
@settings(max_examples=50)
def test_metrics__rates__remain_between_zero_and_one(
    m_n1: float,
    m_n2: float,
    m_n3: float,
    ib1: float,
    ib2: float,
    ib3: float,
    cb1: float,
    cb2: float,
) -> None:
    """METRIC-009: rates ∈ [0,1]; Selectivity ∈ [-1,1]."""
    epsilon = 1e-6
    # Force a non-degenerate baseline: q1 Q+, q2 Q-
    baseline = {"q1": 2.0, "q2": -2.0, "q3": 1.0}
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=baseline,
        epsilon=epsilon,
        tie_policy="merge_into_q_minus",
    )
    artifact = freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash="h_model",
        prompt_template_hash="h_prompt",
        order_manifest_hash="h_order",
        dataset_manifest_hash="h_data",
    )
    metrics = compute_behavioral_metrics(
        frozen_partition=artifact,
        current_neutral_margins={"q1": m_n1, "q2": m_n2, "q3": m_n3},
        current_ib_margins={"q1": [ib1], "q2": [ib2], "q3": [ib3]},
        current_cb_margins={"q1": [cb1], "q2": [cb2], "q3": [cb1]},
        epsilon=epsilon,
    )
    for rate in (
        metrics.neutral_accuracy,
        metrics.ftw,
        metrics.cbr,
        metrics.pra_mean,
        metrics.pra_all,
    ):
        assert rate is not None
        assert 0.0 <= rate <= 1.0
    assert metrics.selectivity is not None
    assert -1.0 <= metrics.selectivity <= 1.0
