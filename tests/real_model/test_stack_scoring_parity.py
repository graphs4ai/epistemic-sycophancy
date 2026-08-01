"""RUN-008: vectorized stack scorer ≡ scalar reference on pinned tiny GPT-2."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.stack.scoring import score_batch_with_lm_logits
from tests.real_model._pin import ATOL, MODEL_ID, MODEL_REVISION, RTOL
from tests.unit.test_stack_scoring import score_batch_with_lm_logits_reference

# Unequal lengths so padding is exercised on a real tokenizer batch.
PROMPTS = (
    "Answer:",
    "Q: sky color? Answer:",
    "Short?",
)
TRUTHFUL_LABELS = ("A", "B", "A")
# DEC-010 ascii letter A/B token ids on GPT-2 BPE.
CONTINUATION_A = (65,)
CONTINUATION_B = (66,)


@pytest.mark.real_model
@pytest.mark.slow
def test_stack_scoring__vectorized__matches_reference_on_real_model_smoke_batch() -> None:
    """RUN-008: optimized scorer matches pre-vectorization reference on tiny GPT-2."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model.eval()
    # Configurations for causal LMs enable KV caching by default.
    assert getattr(model.config, "use_cache", None) is True

    device = torch.device("cpu")
    common = dict(
        model=model,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        continuation_token_ids_A=CONTINUATION_A,
        continuation_token_ids_B=CONTINUATION_B,
        truthful_labels=TRUTHFUL_LABELS,
        device=device,
    )
    optimized = score_batch_with_lm_logits(**common)
    reference = score_batch_with_lm_logits_reference(**common)

    assert optimized.score_a == pytest.approx(reference.score_a, abs=ATOL, rel=RTOL)
    assert optimized.score_b == pytest.approx(reference.score_b, abs=ATOL, rel=RTOL)
    assert optimized.margins == pytest.approx(reference.margins, abs=ATOL, rel=RTOL)
    assert optimized.truthful_labels == reference.truthful_labels
