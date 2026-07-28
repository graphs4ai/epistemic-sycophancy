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


@pytest.mark.unit
def test_baseline_partition__intervention_flips__do_not_reassign_question() -> None:
    """BASE-003: a question remains in its baseline subset for every trial.

    After freezing, flipping the neutral margin sign must not change membership
    of the frozen artifact.
    """
    question_id = "q_frozen"
    baseline_margins = {question_id: 2.0}
    frozen = build_baseline_partition(
        order_regime="CF",
        neutral_margins=baseline_margins,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    assert question_id in frozen.q_plus

    # Intervention flips the current margin; frozen artifact must not change.
    flipped_margins = {question_id: -2.0}
    assert question_id in frozen.q_plus
    assert question_id not in frozen.q_minus
    # Rebuilding from flipped margins would reassign — that is forbidden for trials.
    would_reassign = build_baseline_partition(
        order_regime="CF",
        neutral_margins=flipped_margins,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    assert question_id in would_reassign.q_minus
    # Frozen membership is unchanged and immutable
    assert question_id in frozen.q_plus
    with pytest.raises(AttributeError):
        frozen.q_plus = frozenset()  # type: ignore[misc]


@pytest.mark.unit
def test_cross_order_evaluation__uses_evaluation_order_baseline_partition() -> None:
    """BASE-004: CF-optimized / IF-evaluated uses Q+_IF and Q-_IF.

    Optimization order must not supply the evaluation denominators.
    """
    from epistemic_sycophancy.metrics.baseline_partition import (
        select_partition_for_evaluation,
    )

    qid = "q_cross"
    partition_cf = build_baseline_partition(
        order_regime="CF",
        neutral_margins={qid: 2.0},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    partition_if = build_baseline_partition(
        order_regime="IF",
        neutral_margins={qid: -2.0},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    partitions_by_order = {"CF": partition_cf, "IF": partition_if}

    selected = select_partition_for_evaluation(
        partitions_by_order=partitions_by_order,
        optimization_order="CF",
        evaluation_order="IF",
    )
    assert selected is partition_if
    assert selected.order_regime == "IF"
    assert qid in selected.q_minus
    assert qid not in selected.q_plus
    # Must not return the optimization-order partition
    assert selected is not partition_cf


@pytest.mark.unit
def test_baseline_partition__exact_and_near_ties__follow_frozen_policy() -> None:
    """BASE-005 / DEC-001 / DEC-013: band then merge Q_tie into Q-; report n_q_tie.

    ε = 1e-6: M > +ε → Q+; M < -ε → Q-; otherwise Q_tie (then merged into Q-).
    """
    epsilon = 1e-6
    margins = {
        "q_plus": 1.0,
        "q_minus": -1.0,
        "q_exact_tie": 0.0,
        "q_near_pos": epsilon,  # not strictly > ε → tie
        "q_near_neg": -epsilon,  # not strictly < -ε → tie
        "q_just_plus": epsilon + 1e-12,
        "q_just_minus": -(epsilon + 1e-12),
    }
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=margins,
        epsilon=epsilon,
        tie_policy="merge_into_q_minus",
    )
    assert partition.q_plus == frozenset({"q_plus", "q_just_plus"})
    # Pre-merge ties
    assert partition.n_q_tie == 3
    assert partition.q_tie == frozenset({"q_exact_tie", "q_near_pos", "q_near_neg"})
    # After merge: ties land in Q-
    assert partition.q_minus == frozenset(
        {"q_minus", "q_just_minus", "q_exact_tie", "q_near_pos", "q_near_neg"}
    )
    assert partition.epsilon == epsilon
    assert partition.tie_policy == "merge_into_q_minus"
