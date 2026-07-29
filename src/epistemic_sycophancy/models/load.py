"""Pinned Hugging Face model + tokenizer loading (Phase K RUN-002)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from epistemic_sycophancy.models.spec import ModelSpec


def ensure_cuda_toolkit_include_path() -> None:
    """Prepend system CUDA headers so Triton can compile cuda_utils (Gemma-3 RoPE)."""
    candidates = (
        Path("/usr/local/cuda/include"),
        Path("/usr/local/cuda-13/include"),
        Path("/usr/local/cuda-13.0/include"),
    )
    for include_dir in candidates:
        if (include_dir / "cuda.h").is_file():
            current = os.environ.get("CPATH", "")
            prefix = str(include_dir)
            if prefix not in current.split(":"):
                os.environ["CPATH"] = (
                    prefix if not current else f"{prefix}:{current}"
                )
            return


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

    if spec.device_policy in {"cuda_required", "cuda_if_available"}:
        ensure_cuda_toolkit_include_path()

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
