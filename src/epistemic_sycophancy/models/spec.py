"""Model identity and load-policy specs (Phase K)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Pinned Hugging Face causal-LM identity for an experiment stack."""

    hf_id: str
    revision: str
    tokenizer_revision: str
    dtype: str
    device_policy: str

    def __post_init__(self) -> None:
        if not self.hf_id:
            raise ValueError("hf_id must be a non-empty string")
        if not self.revision:
            raise ValueError("revision must be a non-empty string")
        if not self.tokenizer_revision:
            raise ValueError("tokenizer_revision must be a non-empty string")
        if not self.dtype:
            raise ValueError("dtype must be a non-empty string")
        if not self.device_policy:
            raise ValueError("device_policy must be a non-empty string")
