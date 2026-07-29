"""Frozen MC0 continuation-string / tokenizer contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class InvalidContinuationContract(Exception):
    """Raised when continuation contract fields violate DEC-010."""


@dataclass(frozen=True, slots=True)
class ContinuationContract:
    """Versioned A/B continuation strings and tokenizer pin (DEC-010)."""

    continuation_A: str
    continuation_B: str
    continuation_include_eos: bool
    tokenizer_name: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        if self.continuation_A is None or self.continuation_B is None:
            raise InvalidContinuationContract("continuation strings must be explicit")
        if self.continuation_include_eos is None:
            raise InvalidContinuationContract(
                "continuation_include_eos must be explicit"
            )
        if self.continuation_A != "A" or self.continuation_B != "B":
            raise InvalidContinuationContract(
                "DEC-010 requires exact continuation strings 'A' and 'B'; "
                f"got {self.continuation_A!r} and {self.continuation_B!r}"
            )
        if self.continuation_include_eos:
            raise InvalidContinuationContract(
                "DEC-010 forbids appending EOS to scored continuations"
            )
        if self.tokenizer_name is None or self.tokenizer_revision is None:
            raise InvalidContinuationContract("tokenizer pin must be explicit")


class _AsciiLetterTokenizer:
    """Local pinned Phase B tokenizer: each character maps to its Unicode code point."""

    def __init__(self, *, revision: str) -> None:
        if revision != "v1":
            raise ValueError(f"unsupported ascii_letter revision: {revision!r}")
        self.revision = revision
        self.name = "epistemic_sycophancy.ascii_letter"

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


class SupportsEncode(Protocol):
    def encode(self, text: str) -> list[int]: ...


def load_ascii_letter_tokenizer(*, revision: str) -> _AsciiLetterTokenizer:
    """Load the Phase B unit-pinned ASCII-letter tokenizer (DEC-010)."""
    return _AsciiLetterTokenizer(revision=revision)


def encode_continuations(
    contract: ContinuationContract,
    tokenizer: SupportsEncode,
) -> dict[str, list[int]]:
    """Encode frozen A/B continuations; never append EOS (DEC-010)."""
    if contract.continuation_include_eos:
        raise InvalidContinuationContract(
            "encode_continuations refuses continuation_include_eos=True"
        )
    return {
        "A": list(tokenizer.encode(contract.continuation_A)),
        "B": list(tokenizer.encode(contract.continuation_B)),
    }



def encode_continuation_token_ids(
    *,
    continuation: str,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> list[int]:
    """Encode one continuation string under a pinned tokenizer (DEC-010/043)."""
    if continuation not in {"A", "B"}:
        raise InvalidContinuationContract(
            f"DEC-010 continuations must be 'A' or 'B'; got {continuation!r}"
        )
    if tokenizer_name == "epistemic_sycophancy.ascii_letter":
        tokenizer = load_ascii_letter_tokenizer(revision=tokenizer_revision)
        return list(tokenizer.encode(continuation))
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, revision=tokenizer_revision
    )
    return list(tokenizer.encode(continuation, add_special_tokens=False))
