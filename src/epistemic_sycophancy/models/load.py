"""Pinned Hugging Face model + tokenizer loading (Phase K RUN-002)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from epistemic_sycophancy.models.spec import ModelSpec


@dataclass(frozen=True)
class LoadedModel:
    """Frozen-in-memory model and tokenizer for an InterventionStack."""

    model_id: str
    revision: str
    tokenizer_revision: str
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name!r}")
    return mapping[name]


def _resolve_device(device_policy: str) -> torch.device:
    if device_policy == "cuda_required":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device_policy='cuda_required' but CUDA is unavailable "
                "(DEC-049 / DEC-047: use test-cuda)"
            )
        return torch.device("cuda")
    if device_policy == "cpu":
        return torch.device("cpu")
    if device_policy == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device_policy: {device_policy!r}")


def load_model(spec: ModelSpec) -> LoadedModel:
    """Load a pinned causal LM and tokenizer (DEC-049). Weights are frozen."""
    import transformers

    dtype = _resolve_dtype(spec.dtype)
    device = _resolve_device(spec.device_policy)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec.hf_id,
        revision=spec.tokenizer_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return LoadedModel(
        model_id=spec.hf_id,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
    )
