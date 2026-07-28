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


def score_masked_conditional_log_probs(
    *,
    log_probs: list[float],
    is_pad: list[bool],
) -> float:
    """Sum conditional log-probs excluding pad positions (SCORE-009 / DEC-011)."""
    if len(log_probs) != len(is_pad):
        raise ValueError(
            f"log_probs length {len(log_probs)} != is_pad length {len(is_pad)}"
        )
    kept = [float(lp) for lp, pad in zip(log_probs, is_pad) if not pad]
    if not kept:
        raise ValueError("no non-pad log-probs to score")
    return float(sum(kept))


def score_candidates_batched(
    rows: list[dict],
) -> list[float]:
    """Score a batch of candidate log-prob rows; must match scalar masked sums.

    Pads each row's log-prob / mask sequences to a common max length, then
    sums non-pad positions (DEC-011 sum_log_probs + SCORE-009 masking).
    """
    if not rows:
        return []
    max_len = max(len(row["log_probs"]) for row in rows)
    scores: list[float] = []
    for row in rows:
        log_probs = list(row["log_probs"])
        is_pad = list(row["is_pad"])
        if len(log_probs) != len(is_pad):
            raise ValueError(
                f"row label={row.get('label')!r}: log_probs/is_pad length mismatch"
            )
        # Right-pad batch slots as pad so they never contribute.
        pad_count = max_len - len(log_probs)
        if pad_count:
            log_probs.extend([0.0] * pad_count)
            is_pad.extend([True] * pad_count)
        scores.append(
            score_masked_conditional_log_probs(log_probs=log_probs, is_pad=is_pad)
        )
    return scores
