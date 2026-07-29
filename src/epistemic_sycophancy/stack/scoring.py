"""Score A/B continuations via LM logits through library margins (RUN-008)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from epistemic_sycophancy.scoring.candidates import score_single_token_candidate
from epistemic_sycophancy.scoring.margins import truthful_margin


@dataclass(frozen=True)
class StackScoreBatch:
    """Per-prompt A/B scores and semantic truthful margins."""

    score_a: tuple[float, ...]
    score_b: tuple[float, ...]
    margins: tuple[float, ...]
    truthful_labels: tuple[str, ...]


def score_batch_with_lm_logits(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
    device: torch.device,
) -> StackScoreBatch:
    """Score DEC-010 A/B via next-token LM logits (not a residual linear head).

    Single-token continuations use ``score_single_token_candidate`` on the
    prompt-final logit row. Multi-token IDs are rejected until a dedicated path
    is added.
    """
    if len(prompts) != len(truthful_labels):
        raise ValueError("prompts and truthful_labels must have equal length")
    if len(continuation_token_ids_A) != 1 or len(continuation_token_ids_B) != 1:
        raise ValueError(
            "RUN-008 requires single-token A/B continuations; "
            f"got A={list(continuation_token_ids_A)!r}, "
            f"B={list(continuation_token_ids_B)!r}"
        )
    token_a = int(continuation_token_ids_A[0])
    token_b = int(continuation_token_ids_B[0])

    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
        )
        logits = outputs.logits  # [B, T, V]

    attention = encoded.get("attention_mask")
    scores_a: list[float] = []
    scores_b: list[float] = []
    margins: list[float] = []
    for batch_index, label in enumerate(truthful_labels):
        if attention is not None:
            prompt_length = int(attention[batch_index].sum().item())
        else:
            prompt_length = int(encoded["input_ids"].shape[1])
        row_logits = logits[batch_index, :prompt_length, :].detach().cpu().tolist()
        score_a = score_single_token_candidate(
            row_logits, token_id=token_a, prompt_length=prompt_length
        )
        score_b = score_single_token_candidate(
            row_logits, token_id=token_b, prompt_length=prompt_length
        )
        scores_a.append(score_a)
        scores_b.append(score_b)
        margins.append(
            truthful_margin(score_a=score_a, score_b=score_b, truthful_label=label)
        )
    return StackScoreBatch(
        score_a=tuple(scores_a),
        score_b=tuple(scores_b),
        margins=tuple(margins),
        truthful_labels=tuple(truthful_labels),
    )
