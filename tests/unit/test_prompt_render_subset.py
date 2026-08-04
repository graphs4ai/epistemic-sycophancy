"""WIRE-005: MC0 coverage subset rendering from frozen corpus rows."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.study import StudyFsCoverageConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.prompts.render import (
    render_mc0_subset,
    select_coverage_question_ids,
)


@pytest.mark.unit
def test_prompts__render_mc0_subset__from_frozen_corpus_and_fs_coverage() -> None:
    """WIRE-005: fs_coverage N/seed selects FS IDs and renders deterministic MC0 text."""
    coverage = StudyFsCoverageConfig(n_questions=2, seed=0)
    split_ids = {
        "feature_selection": ("q3", "q1", "q2", "q4"),
        "optimization": ("q10", "q11"),
    }
    selected = select_coverage_question_ids(
        coverage=coverage, split_question_ids=split_ids
    )
    assert len(selected) == 2
    assert set(selected).issubset(set(split_ids["feature_selection"]))
    assert "q10" not in selected

    corpus = [
        {
            "question_id": qid,
            "split": "feature_selection",
            "order_regime": "CF",
            "belief_condition": "N",
            "question_text": f"Q text {qid}?",
            "candidate_a": "Yes",
            "candidate_b": "No",
            "instruction": "Answer with A or B.",
            "suffix": "",
            "truthful_label": "A",
            "format": "MC0",
            "prompt_template_version": "v1",
            "belief_context": None,
        }
        for qid in split_ids["feature_selection"]
    ]
    rows = render_mc0_subset(
        corpus_rows=corpus,
        coverage=coverage,
        split_question_ids=split_ids,
        order_regime="CF",
        belief_condition="N",
    )
    assert len(rows) == 2
    assert {r.question_id for r in rows} == set(selected)
    for row in rows:
        assert "Question:" in row.text
        assert "A. Yes" in row.text
        assert "B. No" in row.text
        assert "Answer with A or B." in row.text


@pytest.mark.unit
def test_prompts__render_mc0_subset__behavior_validation__allowed_for_eval() -> None:
    """DEC-069 eval path: explicit QIDs may render behavior_validation rows; holdout still forbidden."""
    split_ids = {
        "feature_selection": ("q_fs",),
        "optimization": ("q_opt",),
        "behavior_validation": ("q_val",),
        "holdout_test_behavior": ("q_hold",),
    }
    val_row = {
        "question_id": "q_val",
        "split": "behavior_validation",
        "order_regime": "CF",
        "belief_condition": "N",
        "question_text": "Validation question?",
        "candidate_a": "Yes",
        "candidate_b": "No",
        "instruction": "Answer with A or B.",
        "suffix": "",
        "truthful_label": "A",
        "format": "MC0",
        "prompt_template_version": "v1",
        "belief_context": None,
    }
    hold_row = {
        **val_row,
        "question_id": "q_hold",
        "split": "holdout_test_behavior",
        "question_text": "Holdout question?",
    }

    rendered = render_mc0_subset(
        corpus_rows=(val_row,),
        split_question_ids=split_ids,
        order_regime="CF",
        question_ids=("q_val",),
        belief_condition="N",
    )
    assert len(rendered) == 1
    assert rendered[0].question_id == "q_val"
    assert rendered[0].split == "behavior_validation"
    assert "Validation question?" in rendered[0].text

    with pytest.raises(HoldoutAccessError, match="holdout_test_behavior"):
        render_mc0_subset(
            corpus_rows=(hold_row,),
            split_question_ids=split_ids,
            order_regime="CF",
            question_ids=("q_hold",),
            belief_condition="N",
        )
