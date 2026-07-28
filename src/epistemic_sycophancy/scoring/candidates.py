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


def score_multi_token_candidate(
    conditional_log_probs: list[float],
    *,
    aggregation: str,
) -> float:
    """Aggregate per-token conditional log-probs into a candidate score (DEC-011).

    For ``aggregation="sum_log_probs"``:
    s = sum_i log p(t_i | prompt, t_<i).
    """
    if aggregation != "sum_log_probs":
        raise ValueError(
            f"unsupported multi_token_candidate_scoring: {aggregation!r}; "
            "DEC-011 requires 'sum_log_probs'"
        )
    if not conditional_log_probs:
        raise ValueError("conditional_log_probs must be non-empty")
    return float(sum(float(lp) for lp in conditional_log_probs))
