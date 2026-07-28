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


@pytest.mark.unit
def test_baseline_partition__ignores_belief_conditioned_and_intervened_margins() -> None:
    """BASE-002: partition uses unmodified neutral baseline margins only.

    Distractor IB and intervened margins that would flip the subset must not
    affect assignment when only neutral margins are the partition input.
    """
    question_id = "q1"
    # Neutral: clearly Q+
    neutral_margins = {question_id: 2.0}
    # Distractors that would place q1 in Q- if mistakenly used
    belief_conditioned_ib_margins = {question_id: -5.0}
    intervened_margins = {question_id: -3.0}

    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=neutral_margins,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
        # Callers may pass distractors; they must be ignored
        belief_conditioned_margins=belief_conditioned_ib_margins,
        intervened_margins=intervened_margins,
    )
    assert question_id in partition.q_plus
    assert question_id not in partition.q_minus
    # Sanity: distractors alone would have been Q-
    distractor_if_used = build_baseline_partition(
        order_regime="CF",
        neutral_margins=belief_conditioned_ib_margins,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    assert question_id in distractor_if_used.q_minus
