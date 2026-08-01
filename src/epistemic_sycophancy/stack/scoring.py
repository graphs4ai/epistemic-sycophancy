"""Score A/B continuations via LM logits through library margins (RUN-008)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

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

    Single-token continuations read the prompt-final (last non-pad) logit row
    for tokens A and B. Multi-token IDs are rejected until a dedicated path
    is added. Always forwards with ``use_cache=False``.
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

    encoded_cpu = tokenizer(list(prompts), return_tensors="pt", padding=True)

    attention_cpu = encoded_cpu.get("attention_mask")

    if attention_cpu is not None:
        prompt_lengths_cpu = attention_cpu.sum(dim=1, dtype=torch.long)
    else:
        prompt_lengths_cpu = torch.full(
            (len(prompts),),
            encoded_cpu["input_ids"].shape[1],
            dtype=torch.long,
        )

    encoded = {
        key: value.to(device)
        for key, value in encoded_cpu.items()
    }

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=False,
        )

        logits = outputs.logits

        batch_indices = torch.arange(
            logits.shape[0],
            device=logits.device,
        )
        final_positions = (
            prompt_lengths_cpu.to(logits.device) - 1
        )

        candidate_ids = torch.tensor(
            [token_a, token_b],
            dtype=torch.long,
            device=logits.device,
        )

        candidate_logits = logits[
            batch_indices,
            final_positions,
        ].index_select(
            dim=1,
            index=candidate_ids,
        )

    candidate_logits_cpu = (
        candidate_logits
        .float()
        .cpu()
        .tolist()
    )

    scores_a = tuple(float(row[0]) for row in candidate_logits_cpu)
    scores_b = tuple(float(row[1]) for row in candidate_logits_cpu)

    margins = tuple(
        truthful_margin(
            score_a=score_a,
            score_b=score_b,
            truthful_label=label,
        )
        for score_a, score_b, label in zip(
            scores_a,
            scores_b,
            truthful_labels,
            strict=True,
        )
    )

    return StackScoreBatch(
        score_a=scores_a,
        score_b=scores_b,
        margins=margins,
        truthful_labels=tuple(truthful_labels),
    )


def score_batch_through_hooks(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
    device: torch.device,
    install_hooks_cm: Any | None = None,
) -> StackScoreBatch:
    """Score A/B under an optional hooks context manager (β=0 ≡ unhooked).

    When ``install_hooks_cm`` is None, behaves identically to
    ``score_batch_with_lm_logits``. Callers pass
    ``stack.install_hooks(... )`` (or equivalent) for hooked scoring.
    """
    if install_hooks_cm is None:
        return score_batch_with_lm_logits(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            continuation_token_ids_A=continuation_token_ids_A,
            continuation_token_ids_B=continuation_token_ids_B,
            truthful_labels=truthful_labels,
            device=device,
        )
    with install_hooks_cm:
        return score_batch_with_lm_logits(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            continuation_token_ids_A=continuation_token_ids_A,
            continuation_token_ids_B=continuation_token_ids_B,
            truthful_labels=truthful_labels,
            device=device,
        )
