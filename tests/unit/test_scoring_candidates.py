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


@pytest.mark.unit
def test_scoring__padding_tokens__do_not_contribute_to_candidate_log_probability() -> None:
    """SCORE-009: pad positions never enter the candidate log-prob sum."""
    from epistemic_sycophancy.scoring.candidates import (
        score_masked_conditional_log_probs,
    )

    # Candidate token log-probs (true content): two tokens.
    candidate_lps = [-0.4, -0.6]
    expected = -1.0

    # Unpadded: mask all True over candidate positions only.
    unpadded = score_masked_conditional_log_probs(
        log_probs=candidate_lps,
        is_pad=[False, False],
    )
    assert unpadded == pytest.approx(expected, abs=1e-12, rel=1e-12)

    # Left-padded: pad log-probs are huge so any leak would change the score.
    left_padded = score_masked_conditional_log_probs(
        log_probs=[99.0, 88.0, -0.4, -0.6],
        is_pad=[True, True, False, False],
    )
    assert left_padded == pytest.approx(expected, abs=1e-12, rel=1e-12)

    # Right-padded: same candidate, pad after.
    right_padded = score_masked_conditional_log_probs(
        log_probs=[-0.4, -0.6, 77.0, 66.0],
        is_pad=[False, False, True, True],
    )
    assert right_padded == pytest.approx(expected, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_scoring__nan_or_infinite_candidate_score__follows_invalid_row_policy() -> None:
    """SCORE-011 / DEC-012: non-finite scores raise InvalidScoreError (fail_trial)."""
    import math

    from epistemic_sycophancy.scoring.candidates import enforce_finite_candidate_score
    from epistemic_sycophancy.scoring.exceptions import InvalidScoreError

    # Finite score is accepted and returned unchanged.
    assert enforce_finite_candidate_score(
        1.5, invalid_row_policy="fail_trial"
    ) == pytest.approx(1.5, abs=1e-12, rel=1e-12)

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(InvalidScoreError):
            enforce_finite_candidate_score(bad, invalid_row_policy="fail_trial")

    # Silent drop / substitution is forbidden: policy must be fail_trial.
    with pytest.raises(ValueError, match="fail_trial"):
        enforce_finite_candidate_score(math.nan, invalid_row_policy="silent_drop")



@pytest.mark.unit
def test_scoring__batched_candidates__matches_scalar_reference() -> None:
    """SCORE-008: batched scores match slow scalar for A/B, lengths, padding, multi-token."""
    from epistemic_sycophancy.scoring.candidates import (
        score_candidates_batched,
        score_masked_conditional_log_probs,
    )

    # Rows cover: A and B; different lengths; left/right pad; multi-token.
    rows = [
        {"label": "A", "log_probs": [-0.1], "is_pad": [False]},  # single-token A
        {"label": "B", "log_probs": [-0.2], "is_pad": [False]},  # single-token B
        {
            "label": "A",
            "log_probs": [-0.3, -0.4, -0.5],
            "is_pad": [False, False, False],
        },  # multi-token, longer prompt content
        {
            "label": "B",
            "log_probs": [9.0, -0.6, -0.7],
            "is_pad": [True, False, False],
        },  # left-padded multi-token
        {
            "label": "A",
            "log_probs": [-0.8, -0.9, 8.0],
            "is_pad": [False, False, True],
        },  # right-padded multi-token
        {
            "label": "B",
            "log_probs": [-1.0, -1.1],
            "is_pad": [False, False],
        },  # different length
    ]

    # Slow scalar reference (not mocked): per-row masked sum.
    scalar_scores = [
        score_masked_conditional_log_probs(
            log_probs=row["log_probs"],
            is_pad=row["is_pad"],
        )
        for row in rows
    ]
    batched_scores = score_candidates_batched(rows)
    assert len(batched_scores) == len(scalar_scores)
    for batched, scalar in zip(batched_scores, scalar_scores):
        assert batched == pytest.approx(scalar, abs=1e-12, rel=1e-12)



