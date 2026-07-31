"""FSC-007: multi-condition FS still gated to feature_selection-split IDs only."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.study import StudyFsCoverageConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.runner.adapters.jacobian import render_fs_multi_condition_rows
from epistemic_sycophancy.runner.feature_selection import (
    run_feature_selection_stage_computed,
)


def _mixed_corpus() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for qid, split in (
        ("q_fs", "feature_selection"),
        ("q_opt", "optimization"),
        ("q_val", "validation"),
        ("q_hold", "holdout_test_behavior"),
    ):
        for belief in ("N", "IB", "CB"):
            rows.append(
                {
                    "question_id": qid,
                    "split": split,
                    "order_regime": "CF",
                    "belief_condition": belief,
                    "question_text": f"Q {qid}?",
                    "candidate_a": "yes",
                    "candidate_b": "no",
                    "truthful_label": "A",
                    "belief_context": None if belief == "N" else f"{belief}-ctx",
                }
            )
    return rows


@pytest.mark.unit
def test_fs_multi_condition__render__never_includes_downstream_split_rows() -> None:
    """FSC-007: IB/CB render on FS coverage IDs must not pull opt/val/holdout."""
    by_condition = render_fs_multi_condition_rows(
        corpus_rows=_mixed_corpus(),
        question_ids=("q_fs",),
        split_question_ids={
            "feature_selection": ("q_fs",),
            "optimization": ("q_opt",),
        },
        order_regime="CF",
    )
    for belief, rows in by_condition.items():
        assert {r.question_id for r in rows} == {"q_fs"}, belief
        assert all(r.split == "feature_selection" for r in rows), belief


@pytest.mark.unit
def test_fs_stage__overlap_with_optimization_ids__raises_holdout_access_error() -> None:
    """FSC-007: extend HoldoutAccessError coverage under multi-condition dispatch."""

    def jacobian_fn(*, order_regime, question_ids, component="resistance"):
        del order_regime, component
        return {(17, 0): 1.0 for _ in question_ids}

    with pytest.raises(HoldoutAccessError, match="optimization/validation/holdout"):
        run_feature_selection_stage_computed(
            order_regime="CF",
            split_name="feature_selection",
            question_ids=("q_fs", "q_opt"),
            jacobian_fn=jacobian_fn,
            freeze_status="unsealed",
            optimization_question_ids=("q_opt", "q_opt2"),
            validation_question_ids=("q_val",),
            holdout_question_ids=("q_hold",),
        )
