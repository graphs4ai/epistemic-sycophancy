"""Candidate log-probability scoring tests (Phase C SCORE)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.prompts.continuations import (
    ContinuationContract,
    encode_continuations,
    load_ascii_letter_tokenizer,
)
from epistemic_sycophancy.scoring.candidates import score_single_token_candidate


@pytest.mark.unit
def test_scoring__single_token_candidates__uses_next_token_logits() -> None:
    """SCORE-006: score is the logit at the position after the frozen prompt suffix."""
    # Prompt tokens occupy indices 0..L-1; next-token logits are at index L-1.
    prompt_length = 4
    vocab_size = 80
    # Build logits [seq, vocab] with distinctive values at the scoring row.
    logits: list[list[float]] = [
        [float(i * 0.01 + j * 0.001) for j in range(vocab_size)]
        for i in range(prompt_length)
    ]
    # Poison other rows so a wrong position cannot accidentally match.
    for j in range(vocab_size):
        logits[0][j] = -999.0
        logits[1][j] = -888.0
        logits[2][j] = -777.0

    contract = ContinuationContract(
        continuation_A="A",
        continuation_B="B",
        continuation_include_eos=False,
        tokenizer_name="epistemic_sycophancy.ascii_letter",
        tokenizer_revision="v1",
    )
    tokenizer = load_ascii_letter_tokenizer(revision="v1")
    token_ids = encode_continuations(contract, tokenizer)
    assert token_ids["A"] == [65]
    assert token_ids["B"] == [66]

    scoring_row = prompt_length - 1  # exact next-token position after suffix
    logits[scoring_row][65] = 3.25
    logits[scoring_row][66] = -1.5

    score_a = score_single_token_candidate(
        logits,
        token_id=token_ids["A"][0],
        prompt_length=prompt_length,
    )
    score_b = score_single_token_candidate(
        logits,
        token_id=token_ids["B"][0],
        prompt_length=prompt_length,
    )
    assert score_a == pytest.approx(3.25, abs=1e-12, rel=1e-12)
    assert score_b == pytest.approx(-1.5, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_scoring__multi_token_candidates__sums_conditional_log_probabilities() -> None:
    """SCORE-007 / DEC-011: s = sum_i log p(t_i | prompt, t_<i); no length-norm."""
    from epistemic_sycophancy.scoring.candidates import score_multi_token_candidate

    # Independent hand values: sum must be used, not mean.
    conditional_log_probs = [-0.5, -1.0, -0.25]
    expected_sum = -0.5 + -1.0 + -0.25  # -1.75
    expected_mean = expected_sum / 3.0

    score = score_multi_token_candidate(
        conditional_log_probs,
        aggregation="sum_log_probs",
    )
    assert score == pytest.approx(expected_sum, abs=1e-12, rel=1e-12)
    assert score != pytest.approx(expected_mean, abs=1e-12, rel=1e-12)

