"""RUN-009 / WIRE-006: baseline partition stage."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.runner.stages import (
    run_baseline_partition_stage,
    run_baseline_partition_stage_via_scores,
)


@pytest.mark.unit
def test_runner__baseline_partition_stage__uses_fs_split_and_rejects_holdout() -> None:
    """RUN-009: FS-only neutral margins; holdout access raises HoldoutAccessError."""
    fs_neutral = {"q1": 1.0, "q2": -0.5, "q3": 0.0}
    partition = run_baseline_partition_stage(
        split_name="feature_selection",
        order_regime="CF",
        neutral_margins=fs_neutral,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
        freeze_status="unsealed",
    )
    assert partition.order_regime == "CF"
    assert "q1" in partition.q_plus
    assert "q2" in partition.q_minus

    with pytest.raises(HoldoutAccessError):
        run_baseline_partition_stage(
            split_name="holdout_test_behavior",
            order_regime="CF",
            neutral_margins=fs_neutral,
            epsilon=1e-6,
            tie_policy="merge_into_q_minus",
            freeze_status="unsealed",
        )


@pytest.mark.unit
def test_runner__baseline_partition_stage__scores_fs_subset_via_stack() -> None:
    """WIRE-006: score_fn supplies neutral margins; FS only; holdout sealed."""
    scored: list[str] = []

    def score_fn(question_ids):
        scored.extend(question_ids)
        return {qid: (1.0 if qid == "q1" else -0.5) for qid in question_ids}

    partition = run_baseline_partition_stage_via_scores(
        split_name="feature_selection",
        order_regime="CF",
        question_ids=("q1", "q2"),
        score_fn=score_fn,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
        freeze_status="unsealed",
    )
    assert scored == ["q1", "q2"]
    assert "q1" in partition.q_plus
    assert "q2" in partition.q_minus

    with pytest.raises(HoldoutAccessError):
        run_baseline_partition_stage_via_scores(
            split_name="holdout_test_behavior",
            order_regime="CF",
            question_ids=("q1",),
            score_fn=score_fn,
            epsilon=1e-6,
            tie_policy="merge_into_q_minus",
            freeze_status="unsealed",
        )
