"""RUN-008: InterventionStack score_batch ≡ library truthful margins."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from epistemic_sycophancy.scoring.candidates import score_single_token_candidate
from epistemic_sycophancy.scoring.margins import truthful_margin
from epistemic_sycophancy.stack.scoring import score_batch_with_lm_logits


class _ToyCausalLM(nn.Module):
    """Minimal causal LM: fixed last-position logits over a tiny vocab."""

    def __init__(self, vocab_logits: torch.Tensor) -> None:
        super().__init__()
        self.vocab_logits = vocab_logits  # [vocab]
        self.device = torch.device("cpu")

    def __call__(self, *, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        vocab = self.vocab_logits.shape[0]
        logits = torch.zeros(batch, seq, vocab, dtype=torch.float64)
        logits[:, -1, :] = self.vocab_logits
        return SimpleNamespace(logits=logits)


@pytest.mark.unit
def test_stack__score_batch__truthful_margins_match_library_api() -> None:
    """RUN-008: LM next-token A/B scores → truthful_margin (not linear residual head)."""
    # Vocab: id 0='A', id 1='B', others filler. Last-token logits favor A.
    vocab_logits = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    model = _ToyCausalLM(vocab_logits)
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: {"A": [0], "B": [1]}[text]
    )
    prompts = ["prompt one", "prompt two"]
    # Fake tokenizer that maps prompts to length-3 id sequences.
    class _Tok:
        def __call__(self, texts, return_tensors="pt", padding=True):
            batch = len(texts)
            return {
                "input_ids": torch.zeros(batch, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch, 3, dtype=torch.long),
            }

        def encode(self, text, add_special_tokens=False):
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    results = score_batch_with_lm_logits(
        model=model,
        tokenizer=_Tok(),
        prompts=prompts,
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=("A", "B"),
        device=torch.device("cpu"),
    )
    assert len(results.margins) == 2

    # Independent library computation from the same fixed last-token logits.
    logits_rows = vocab_logits.tolist()
    # score_single_token_candidate expects [seq, vocab]; use a 3-row pad with last row.
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
