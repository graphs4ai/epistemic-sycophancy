"""Single- and multi-token candidate scoring from logits / log-probs."""

from __future__ import annotations


def score_single_token_candidate(
    logits: list[list[float]],
    *,
    token_id: int,
    prompt_length: int,
) -> float:
    """Return the next-token logit for ``token_id`` after a prompt of given length.

    ``logits`` has shape ``[seq_len, vocab]``. The scoring position is
    ``prompt_length - 1`` — the row that predicts the token immediately after
    the frozen prompt suffix.
    """
    if prompt_length < 1:
        raise ValueError(f"prompt_length must be >= 1; got {prompt_length}")
    if len(logits) < prompt_length:
        raise ValueError(
            f"logits length {len(logits)} shorter than prompt_length {prompt_length}"
        )
    scoring_position = prompt_length - 1
    return float(logits[scoring_position][token_id])
