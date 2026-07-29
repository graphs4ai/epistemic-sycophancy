"""RUN-002: pinned Gemma-3-4B-IT tokenizer continuations vs DEC-010."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.models.load import load_model
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.prompts.continuations import encode_continuation_token_ids

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "real_model"
    / "gemma3_continuation_token_ids.json"
)


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_models__load_pinned_gemma__tokenizer_continuations_match_dec010() -> None:
    """RUN-002: pinned Gemma revision; A/B token IDs match DEC-010/049 fixture."""
    golden = json.loads(_GOLDEN.read_text())
    assert golden["continuation_A"] == "A"
    assert golden["continuation_B"] == "B"
    assert golden["continuation_include_eos"] is False

    spec = ModelSpec(
        hf_id=golden["model_id"],
        revision=golden["revision"],
        tokenizer_revision=golden["tokenizer_revision"],
        dtype="bfloat16",
        device_policy="cuda_required",
    )
    loaded = load_model(spec)
    assert loaded.model_id == golden["model_id"]
    assert loaded.revision == golden["revision"]
    assert loaded.tokenizer_revision == golden["tokenizer_revision"]
    assert loaded.tokenizer is not None
    assert loaded.model is not None

    ids_a = encode_continuation_token_ids(
        continuation="A",
        tokenizer_name=golden["model_id"],
        tokenizer_revision=golden["tokenizer_revision"],
    )
    ids_b = encode_continuation_token_ids(
        continuation="B",
        tokenizer_name=golden["model_id"],
        tokenizer_revision=golden["tokenizer_revision"],
    )
    assert ids_a == golden["token_ids_A"]
    assert ids_b == golden["token_ids_B"]

    # Tokenizer from load_model must agree with the pin (DEC-010 strings unchanged).
    tok = loaded.tokenizer
    assert list(tok.encode("A", add_special_tokens=False)) == golden["token_ids_A"]
    assert list(tok.encode("B", add_special_tokens=False)) == golden["token_ids_B"]
