"""ADAPT-001: processed MC0 corpus bridge + coverage/optimize QID resolution (DEC-078)."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.config.study import StudyOptimizeConfig, StudyFsCoverageConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.prompts.render import render_mc0_subset

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "adapters"
PROCESSED_JSONL = FIXTURE_ROOT / "processed_mc0_tiny.jsonl"
SPLIT_MANIFEST = FIXTURE_ROOT / "split_manifest_tiny.csv"


@pytest.mark.unit
def test_adapters__load_processed_mc0__normalizes_belief_and_order_for_render() -> None:
    """ADAPT-001: processed jsonl maps to CF/IF + N/CB/IB; RO via DEC-009; holdout sealed."""
    from epistemic_sycophancy.runner.adapters.corpus import (
        load_processed_mc0_corpus,
        resolve_optimize_coverage_ids,
        resolve_fs_coverage_question_ids,
        split_question_ids_from_manifest,
    )

    corpus = load_processed_mc0_corpus(
        jsonl_paths=(PROCESSED_JSONL,),
        ro_seed=42,
    )
    # CF/IF neutrals present; beliefs normalized.
    by_key = {
        (r["question_id"], r["order_regime"], r["belief_condition"]): r for r in corpus
    }
    assert ("q_fs_1", "CF", "N") in by_key
    assert ("q_fs_1", "IF", "N") in by_key
    assert ("q_fs_1", "CF", "CB") in by_key
    assert ("q_fs_1", "CF", "IB") in by_key
    # RO synthesized from CF/IF via DEC-009.
    assert ("q_fs_1", "RO", "N") in by_key
    assert by_key[("q_fs_1", "CF", "N")]["candidate_a"] == "Truth A1"
    assert by_key[("q_fs_1", "CF", "N")]["candidate_b"] == "False B1"
    assert by_key[("q_fs_1", "CF", "N")]["truthful_label"] == "A"
    assert by_key[("q_fs_1", "IF", "N")]["truthful_label"] == "B"
    # Holdout rows must not appear in the normalized corpus.
    assert all(r["split"] != "holdout_test_behavior" for r in corpus)
    assert all(r["question_id"] != "q_hold_1" for r in corpus)

    split_ids = split_question_ids_from_manifest(SPLIT_MANIFEST)
    assert "holdout_test_behavior" not in split_ids or "q_hold_1" in split_ids.get(
        "holdout_test_behavior", ()
    )
    # Coverage resolution never returns holdout IDs.
    coverage = StudyFsCoverageConfig(n_questions=2, seed=0)
    selected = resolve_fs_coverage_question_ids(
        coverage=coverage,
        split_question_ids=split_ids,
    )
    assert len(selected) == 2
    assert set(selected) <= {"q_fs_1", "q_fs_2"}
    assert "q_hold_1" not in selected

    # render_mc0_subset accepts normalized rows.
    rendered = render_mc0_subset(
        corpus_rows=corpus,
        coverage=coverage,
        split_question_ids=split_ids,
        order_regime="CF",
        belief_condition="N",
    )
    assert len(rendered) == 2
    assert {r.question_id for r in rendered} == set(selected)

    # Optimize coverage (DEC-068 n_questions).
    opt = StudyOptimizeConfig(
        budget_match_on="n_objective_evals",
        max_steps=20,
        n_questions=3,
    )
    opt_ids = resolve_optimize_coverage_ids(
        optimize=opt,
        split_question_ids=split_ids,
    )
    assert len(opt_ids) == 3
    assert set(opt_ids) <= set(split_ids["optimization"])

    # Requesting holdout split explicitly is forbidden until holdout_eval (DEC-078).
    with pytest.raises(HoldoutAccessError):
        load_processed_mc0_corpus(
            jsonl_paths=(PROCESSED_JSONL,),
            ro_seed=42,
            splits=("holdout_test_behavior",),
        )
