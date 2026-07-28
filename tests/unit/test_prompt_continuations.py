"""Continuation-string / tokenizer contract tests (Phase B)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.prompts.continuations import (
    ContinuationContract,
    encode_continuations,
    load_ascii_letter_tokenizer,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "prompts"
    / "continuation_token_ids_ascii_letter_v1.json"
)


@pytest.mark.unit
def test_prompt__answer_continuations__match_frozen_tokenizer_contract() -> None:
    """PROMPT-007: frozen A/B strings match versioned tokenizer token IDs (DEC-010)."""
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    # Independent expectations: Unicode code points for bare letters.
    assert fixture["token_ids"]["A"] == [ord("A")]
    assert fixture["token_ids"]["B"] == [ord("B")]

    contract = ContinuationContract(
        continuation_A="A",
        continuation_B="B",
        continuation_include_eos=False,
        tokenizer_name="epistemic_sycophancy.ascii_letter",
        tokenizer_revision="v1",
    )
    assert contract.continuation_A == "A"
    assert contract.continuation_B == "B"
    assert not contract.continuation_A.startswith(" ")
    assert not contract.continuation_B.startswith(" ")
    assert "\n" not in contract.continuation_A
    assert "\n" not in contract.continuation_B
    assert contract.continuation_include_eos is False

    tokenizer = load_ascii_letter_tokenizer(revision="v1")
    encoded = encode_continuations(contract, tokenizer)
    assert encoded["A"] == fixture["token_ids"]["A"]
    assert encoded["B"] == fixture["token_ids"]["B"]
    assert encoded["A"] == [65]
    assert encoded["B"] == [66]
