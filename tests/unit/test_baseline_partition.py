"""Baseline partition tests (Phase D BASE)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.metrics.baseline_partition import build_baseline_partition


@pytest.mark.unit
def test_baseline_partition__same_question__may_belong_to_different_subsets_by_order() -> None:
    """BASE-001: CF, IF, and RO partitions are independent artifacts.

    Same question_id with CF M>0 and IF M<0 lands in different subsets.
    """
    question_id = "q_shared"
    # CF: positive neutral margin → Q+
    partition_cf = build_baseline_partition(
        order_regime="CF",
        neutral_margins={question_id: 1.5},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    # IF: negative neutral margin → Q-
    partition_if = build_baseline_partition(
        order_regime="IF",
        neutral_margins={question_id: -1.5},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    assert partition_cf.order_regime == "CF"
    assert partition_if.order_regime == "IF"
    assert question_id in partition_cf.q_plus
    assert question_id not in partition_cf.q_minus
    assert question_id in partition_if.q_minus
    assert question_id not in partition_if.q_plus
    # Independent artifacts (not the same object / not averaged)
    assert partition_cf is not partition_if
