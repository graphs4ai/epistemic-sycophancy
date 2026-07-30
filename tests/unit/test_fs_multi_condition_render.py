"""FSC-001: production FS batch builder renders N, IB, and CB on FS split."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.study import StudySmokeConfig


def _corpus_rows() -> list[dict[str, object]]:
    """Two FS questions with multi-variant IB/CB and duplicate neutrals."""
    rows: list[dict[str, object]] = []
    for qid, truth in (("q_fs_1", "A"), ("q_fs_2", "A")):
        # Two neutral rows (same question) — must dedupe to one.
        for _ in range(2):
            rows.append(
                {
                    "question_id": qid,
                    "split": "feature_selection",
                    "order_regime": "CF",
                    "belief_condition": "N",
                    "question_text": f"Q {qid}?",
                    "candidate_a": "yes",
                    "candidate_b": "no",
                    "truthful_label": truth,
                    "belief_context": None,
                }
            )
        for variant in ("a", "b"):
            rows.append(
                {
                    "question_id": qid,
                    "split": "feature_selection",
                    "order_regime": "CF",
                    "belief_condition": "IB",
                    "question_text": f"Q {qid}?",
                    "candidate_a": "yes",
                    "candidate_b": "no",
                    "truthful_label": truth,
                    "belief_context": f"IB-{variant}",
                }
            )
        for variant in ("a", "b", "c"):
            rows.append(
                {
                    "question_id": qid,
                    "split": "feature_selection",
                    "order_regime": "CF",
                    "belief_condition": "CB",
                    "question_text": f"Q {qid}?",
                    "candidate_a": "yes",
                    "candidate_b": "no",
                    "truthful_label": truth,
                    "belief_context": f"CB-{variant}",
                }
            )
    # Optimization-split rows must not appear in FS render.
    rows.append(
        {
            "question_id": "q_opt_1",
            "split": "optimization",
            "order_regime": "CF",
            "belief_condition": "IB",
            "question_text": "Opt?",
            "candidate_a": "yes",
            "candidate_b": "no",
            "truthful_label": "A",
            "belief_context": "opt-ib",
        }
    )
    return rows


@pytest.mark.unit
def test_fs_batch__renders_all_three_conditions__n_deduped_ib_cb_keep_variants() -> None:
    """FSC-001 / DEC-085: FS render returns N (1/q) + all IB/CB variants on FS IDs."""
    from epistemic_sycophancy.runner.adapters.jacobian import (
        render_fs_multi_condition_rows,
    )

    smoke = StudySmokeConfig(question_ids=("q_fs_1", "q_fs_2"))
    split_ids = {
        "feature_selection": ("q_fs_1", "q_fs_2"),
        "optimization": ("q_opt_1",),
    }
    by_condition = render_fs_multi_condition_rows(
        corpus_rows=_corpus_rows(),
        smoke=smoke,
        split_question_ids=split_ids,
        order_regime="CF",
    )

    assert set(by_condition.keys()) == {"N", "IB", "CB"}

    n_rows = by_condition["N"]
    assert len(n_rows) == 2
    assert {r.question_id for r in n_rows} == {"q_fs_1", "q_fs_2"}
    assert all(r.belief_condition == "N" for r in n_rows)

    ib_rows = by_condition["IB"]
    assert len(ib_rows) == 4  # 2 questions × 2 variants
    assert all(r.belief_condition == "IB" for r in ib_rows)
    assert {r.question_id for r in ib_rows} == {"q_fs_1", "q_fs_2"}

    cb_rows = by_condition["CB"]
    assert len(cb_rows) == 6  # 2 questions × 3 variants
    assert all(r.belief_condition == "CB" for r in cb_rows)
    assert {r.question_id for r in cb_rows} == {"q_fs_1", "q_fs_2"}

    all_qids = {r.question_id for rows in by_condition.values() for r in rows}
    assert "q_opt_1" not in all_qids
