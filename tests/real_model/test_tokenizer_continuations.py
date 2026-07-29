"""REAL-001: pinned experiment tokenizer A/B continuation regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.prompts.continuations import encode_continuation_token_ids
from tests.real_model._pin import MODEL_ID, MODEL_REVISION

_GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "real_model"
    / "continuation_token_ids_tiny_gpt2.json"
)


@pytest.mark.real_model
@pytest.mark.slow
def test_real_model__tokenizer_continuations__match_pinned_ab_token_ids() -> None:
    """REAL-001: exact A/B token IDs under DEC-043 tokenizer (DEC-010 strings)."""
    golden = json.loads(_GOLDEN.read_text())
    assert golden["continuation_A"] == "A"
    assert golden["continuation_B"] == "B"
    assert golden["model_id"] == MODEL_ID
    assert golden["revision"] == MODEL_REVISION

    ids_a = encode_continuation_token_ids(
        continuation="A",
        tokenizer_name=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
    )
    ids_b = encode_continuation_token_ids(
        continuation="B",
        tokenizer_name=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
    )
    assert ids_a == golden["token_ids_A"]
    assert ids_b == golden["token_ids_B"]
