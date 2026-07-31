"""WIRE-005: MC0 coverage subset rendering from frozen corpus rows."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.study import StudyFsCoverageConfig
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
