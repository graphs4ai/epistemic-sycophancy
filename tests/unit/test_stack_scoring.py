"""RUN-008: InterventionStack score_batch ≡ library truthful margins."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from epistemic_sycophancy.scoring.candidates import score_single_token_candidate
from epistemic_sycophancy.scoring.margins import truthful_margin
from epistemic_sycophancy.stack.scoring import (
    StackScoreBatch,
    score_batch_through_hooks,
    score_batch_with_lm_logits,
)


def score_batch_with_lm_logits_reference(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
    device: torch.device,
) -> StackScoreBatch:
    """Pre-vectorization scalar reference for RUN-008 (right-padded batches)."""
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
            use_cache=False,
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


class _ToyCausalLM(nn.Module):
    """Minimal causal LM with per-row final-token logits (right-pad aware)."""

    def __init__(self, row_vocab_logits: torch.Tensor) -> None:
        super().__init__()
        # [B, V] or [V] broadcast across the batch.
        if row_vocab_logits.ndim == 1:
            row_vocab_logits = row_vocab_logits.unsqueeze(0)
        self.row_vocab_logits = row_vocab_logits  # [B, V]
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(use_cache=True)
        self.last_forward_kwargs: dict[str, Any] = {}

    def __call__(self, *, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        batch, seq = input_ids.shape
        vocab = self.row_vocab_logits.shape[1]
        # Poison every position so indexing the pad column cannot accidentally pass.
        logits = torch.full((batch, seq, vocab), -999.0, dtype=torch.float64)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        else:
            lengths = torch.full((batch,), seq, dtype=torch.long)
        for batch_index in range(batch):
            final_pos = int(lengths[batch_index].item()) - 1
            row = self.row_vocab_logits[min(batch_index, self.row_vocab_logits.shape[0] - 1)]
            logits[batch_index, final_pos, :] = row
        return SimpleNamespace(logits=logits)


class _FixedLengthTok:
    """Tokenizer stub that emits fixed equal-length right-padded ids."""

    def __call__(self, texts, return_tensors="pt", padding=True):
        batch = len(texts)
        return {
            "input_ids": torch.zeros(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }


class _VariableLengthTok:
    """Tokenizer stub that right-pads rows to distinct prompt lengths."""

    def __init__(self, lengths: Sequence[int]) -> None:
        self.lengths = tuple(int(length) for length in lengths)

    def __call__(self, texts, return_tensors="pt", padding=True):
        if len(texts) != len(self.lengths):
            raise ValueError("texts/lengths mismatch")
        max_len = max(self.lengths)
        input_ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(texts), max_len, dtype=torch.long)
        for row, length in enumerate(self.lengths):
            attention_mask[row, :length] = 1
            input_ids[row, :length] = row + 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.mark.unit
def test_stack__score_batch__truthful_margins_match_library_api() -> None:
    """RUN-008: LM next-token A/B scores → truthful_margin (not linear residual head)."""
    vocab_logits = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    model = _ToyCausalLM(vocab_logits)
    prompts = ["prompt one", "prompt two"]

    results = score_batch_with_lm_logits(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=prompts,
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A", "B"),
        device=torch.device("cpu"),
    )
    assert len(results.margins) == 2

    logits_rows = vocab_logits.tolist()
    seq_logits = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], logits_rows]
    score_a = score_single_token_candidate(
        seq_logits, token_id=0, prompt_length=3
    )
    score_b = score_single_token_candidate(
        seq_logits, token_id=1, prompt_length=3
    )
    expected_0 = truthful_margin(score_a=score_a, score_b=score_b, truthful_label="A")
    expected_1 = truthful_margin(score_a=score_a, score_b=score_b, truthful_label="B")
    assert results.margins[0] == pytest.approx(expected_0)
    assert results.margins[1] == pytest.approx(expected_1)
    assert results.score_a[0] == pytest.approx(score_a)
    assert results.score_b[0] == pytest.approx(score_b)


@pytest.mark.unit
def test_stack_scoring__beta_zero_hooked__margins_match_unhooked_library() -> None:
    """WIRE-004: β=0 hooked score_batch margins match unhooked library path."""
    vocab_logits = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    model = _ToyCausalLM(vocab_logits)

    unhooked = score_batch_with_lm_logits(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=["p1", "p2"],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A", "B"),
        device=torch.device("cpu"),
    )
    hooked = score_batch_through_hooks(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=["p1", "p2"],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A", "B"),
        device=torch.device("cpu"),
        install_hooks_cm=None,
    )
    assert hooked.margins == unhooked.margins
    assert hooked.score_a == unhooked.score_a


@pytest.mark.unit
def test_stack_scoring__variable_prompt_lengths__uses_per_row_final_nonpad_logits() -> None:
    """RUN-008: padded batch with unequal lengths scores each row at its own t*."""
    # Distinct A/B logits per row; pad columns are poisoned to -999.
    row_logits = torch.tensor(
        [
            [2.5, -1.0, 0.0],
            [0.5, 3.0, 0.0],
            [-2.0, -0.25, 0.0],
        ],
        dtype=torch.float64,
    )
    model = _ToyCausalLM(row_logits)
    lengths = (2, 5, 3)
    prompts = ["aa", "bbbbb", "ccc"]

    results = score_batch_with_lm_logits(
        model=model,
        tokenizer=_VariableLengthTok(lengths),
        prompts=prompts,
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A", "B", "A"),
        device=torch.device("cpu"),
    )

    assert results.score_a == pytest.approx((2.5, 0.5, -2.0))
    assert results.score_b == pytest.approx((-1.0, 3.0, -0.25))
    assert results.margins == pytest.approx(
        (
            truthful_margin(score_a=2.5, score_b=-1.0, truthful_label="A"),
            truthful_margin(score_a=0.5, score_b=3.0, truthful_label="B"),
            truthful_margin(score_a=-2.0, score_b=-0.25, truthful_label="A"),
        )
    )

    reference = score_batch_with_lm_logits_reference(
        model=model,
        tokenizer=_VariableLengthTok(lengths),
        prompts=prompts,
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A", "B", "A"),
        device=torch.device("cpu"),
    )
    assert results.score_a == pytest.approx(reference.score_a)
    assert results.score_b == pytest.approx(reference.score_b)
    assert results.margins == pytest.approx(reference.margins)


@pytest.mark.unit
def test_stack_scoring__truthful_label_A__margin_is_score_a_minus_score_b() -> None:
    """RUN-008: truthful_label A ⇒ M = s_A − s_B."""
    model = _ToyCausalLM(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
    results = score_batch_with_lm_logits(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=["prompt"],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A",),
        device=torch.device("cpu"),
    )
    assert results.score_a[0] == pytest.approx(4.0)
    assert results.score_b[0] == pytest.approx(1.0)
    assert results.margins[0] == pytest.approx(3.0)


@pytest.mark.unit
def test_stack_scoring__truthful_label_B__margin_is_score_b_minus_score_a() -> None:
    """RUN-008: truthful_label B ⇒ M = s_B − s_A."""
    model = _ToyCausalLM(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
    results = score_batch_with_lm_logits(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=["prompt"],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("B",),
        device=torch.device("cpu"),
    )
    assert results.score_a[0] == pytest.approx(4.0)
    assert results.score_b[0] == pytest.approx(1.0)
    assert results.margins[0] == pytest.approx(-3.0)


@pytest.mark.unit
def test_stack_scoring__known_fake_logits__returns_exact_ab_scores() -> None:
    """RUN-008: A/B scores equal the planted final-token logits exactly."""
    model = _ToyCausalLM(torch.tensor([1.25, -3.5, 9.0], dtype=torch.float64))
    results = score_batch_with_lm_logits(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=["p"],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A",),
        device=torch.device("cpu"),
    )
    assert results.score_a == (1.25,)
    assert results.score_b == (-3.5,)
    assert results.margins == (4.75,)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ids_a", "ids_b"),
    [
        ([], [1]),
        ([0], []),
        ([], []),
        ([0, 1], [2]),
        ([0], [1, 2]),
        ([0, 1], [2, 3]),
    ],
)
def test_stack_scoring__empty_or_multitoken_continuation_ids__raises_value_error(
    ids_a: list[int],
    ids_b: list[int],
) -> None:
    """RUN-008: empty or multi-token A/B continuation IDs are rejected."""
    model = _ToyCausalLM(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="single-token A/B continuations"):
        score_batch_with_lm_logits(
            model=model,
            tokenizer=_FixedLengthTok(),
            prompts=["p"],
            continuation_token_ids_A=ids_a,
            continuation_token_ids_B=ids_b,
            truthful_labels=("A",),
            device=torch.device("cpu"),
        )


@pytest.mark.unit
def test_stack_scoring__model_config_use_cache_true__forward_passes_use_cache_false() -> None:
    """RUN-008: disable KV cache even when model.config.use_cache defaults to True."""
    model = _ToyCausalLM(torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64))
    assert model.config.use_cache is True

    score_batch_with_lm_logits(
        model=model,
        tokenizer=_FixedLengthTok(),
        prompts=["p"],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A",),
        device=torch.device("cpu"),
    )
    assert model.last_forward_kwargs.get("use_cache") is False
