"""ORCH-LOG-007c: pre-compute fixed Adam-step prompt-microbatch totals."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.runner.adapters.corpus import _normalize_processed_row
from epistemic_sycophancy.runner.progress import (
    count_adam_step_prompt_microbatches,
    n_prompt_microbatches,
)


def _corpus_three() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, qid in enumerate(("q0", "q1", "q2")):
        for belief in ("neutral", "incorrect", "correct"):
            rows.append(
                {
                    "question_id": qid,
                    "split": "optimization",
                    "format": "mc0",
                    "belief_condition": belief,
                    "answer_order": "true-first",
                    "correct_label": "A",
                    "option_a": f"Truth {i}",
                    "option_b": f"False {i}",
                    "prompt": (
                        f"The user believes the following: I believe that X.\n\n"
                        f"Question: Q{i}?"
                        if belief != "neutral"
                        else f"Question: Q{i}?"
                    ),
                }
            )
    return [_normalize_processed_row(r) for r in rows]


@pytest.mark.unit
def test_n_prompt_microbatches__ceil_division__matches_chunk_count() -> None:
    assert n_prompt_microbatches(0, batch_size=1) == 0
    assert n_prompt_microbatches(3, batch_size=1) == 3
    assert n_prompt_microbatches(3, batch_size=2) == 2
    assert n_prompt_microbatches(4, batch_size=2) == 2


@pytest.mark.unit
def test_count_adam_step_prompt_microbatches__matches_grad_plus_objective_graph() -> None:
    """Fixed total = 5·bN + 3·bIB + 5·bCB (two payloads + one jac)."""
    corpus = _corpus_three()
    split_ids = {"optimization": ("q0", "q1", "q2")}
    qids = ("q0", "q1", "q2")
    # batch_size=1 → bN=bIB=bCB=3 → 5*3 + 3*3 + 5*3 = 39
    total = count_adam_step_prompt_microbatches(
        corpus=corpus,
        split_question_ids=split_ids,
        question_ids=qids,
        order_regime="CF",
        prompt_batch_size=1,
    )
    assert total == 39
    # Uneven chunks: b=2 → ceil(3/2)=2 each → 5*2+3*2+5*2 = 26
    total2 = count_adam_step_prompt_microbatches(
        corpus=corpus,
        split_question_ids=split_ids,
        question_ids=qids,
        order_regime="CF",
        prompt_batch_size=2,
    )
    assert total2 == 26
