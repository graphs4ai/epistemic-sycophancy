"""FSC-002: production FS component prompt selection uses frozen partitions."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.study import StudyFsCoverageConfig
from epistemic_sycophancy.metrics.baseline_partition import BaselinePartition
from epistemic_sycophancy.runner.adapters.jacobian import (
    render_fs_multi_condition_rows,
    select_fs_component_prompt_rows,
)


def _corpus_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for qid in ("q1", "q2", "q3"):
        rows.append(
            {
                "question_id": qid,
                "split": "feature_selection",
                "order_regime": "CF",
                "belief_condition": "N",
                "question_text": f"Q {qid}?",
                "candidate_a": "yes",
                "candidate_b": "no",
                "truthful_label": "A",
                "belief_context": None,
            }
        )
        for belief, n_var in (("IB", 2), ("CB", 2)):
            for v in range(n_var):
                rows.append(
                    {
                        "question_id": qid,
                        "split": "feature_selection",
                        "order_regime": "CF",
                        "belief_condition": belief,
                        "question_text": f"Q {qid}?",
                        "candidate_a": "yes",
                        "candidate_b": "no",
                        "truthful_label": "A",
                        "belief_context": f"{belief}-{v}",
                    }
                )
    return rows


@pytest.mark.unit
@pytest.mark.parametrize(
    ("component", "expected_condition", "expected_questions"),
    [
        ("resistance", "IB", {"q1", "q3"}),
        ("recovery", "CB", {"q2"}),
        ("neutral_surrogate", "N", {"q1", "q2", "q3"}),
        ("correct_surrogate", "CB", {"q1", "q3"}),
    ],
)
def test_fs_adapter__component_prompts__match_frozen_partition_no_cross_condition(
    component: str,
    expected_condition: str,
    expected_questions: set[str],
) -> None:
    """FSC-002 / FEAT-023: production path uses selection_component_prompts."""
    partition = BaselinePartition(
        order_regime="CF",
        q_plus=frozenset({"q1", "q3"}),
        q_minus=frozenset({"q2"}),
        q_tie=frozenset(),
        n_q_tie=0,
    )
    by_condition = render_fs_multi_condition_rows(
        corpus_rows=_corpus_rows(),
        question_ids=("q1", "q2", "q3"),
        split_question_ids={"feature_selection": ("q1", "q2", "q3")},
        order_regime="CF",
    )
    selected = select_fs_component_prompt_rows(
        component=component,
        by_condition=by_condition,
        partition=partition,
    )
    assert {r.belief_condition for r in selected} == {expected_condition}
    assert {r.question_id for r in selected} == expected_questions
    # No cross-condition leakage.
    assert all(r.belief_condition == expected_condition for r in selected)
    foreign = {"N", "IB", "CB"} - {expected_condition}
    assert not any(r.belief_condition in foreign for r in selected)
