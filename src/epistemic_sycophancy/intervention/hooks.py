"""Residual hook token-scope masking (Phase E SAE-010+)."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def build_token_scope_mask(
    *,
    batch_size: int,
    seq_len: int,
    prompt_lengths: Sequence[int],
    token_scope: str,
    k: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return boolean mask [B, T] of positions that may receive Δx (DEC-015).

    Modes:
      - last_prompt_token: only index prompt_length-1
      - all_prompt_tokens: indices [0, prompt_length)
      - last_k_prompt_tokens: last k positions within the prompt (requires k >= 1)

    Positions at or beyond prompt_length (padding / generated answers) stay False.
    """
    if len(prompt_lengths) != batch_size:
        raise ValueError(
            f"prompt_lengths length must equal batch_size; "
            f"got {len(prompt_lengths)} vs {batch_size}"
        )
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for batch_index, prompt_length in enumerate(prompt_lengths):
        if prompt_length <= 0:
            continue
        if token_scope == "last_prompt_token":
            mask[batch_index, prompt_length - 1] = True
        elif token_scope == "all_prompt_tokens":
            mask[batch_index, :prompt_length] = True
        elif token_scope == "last_k_prompt_tokens":
            if k is None or k < 1:
                raise ValueError(
                    "last_k_prompt_tokens requires positive int k; "
                    f"got k={k!r}"
                )
            start = max(0, prompt_length - k)
            mask[batch_index, start:prompt_length] = True
        else:
            raise ValueError(f"unsupported token_scope: {token_scope!r}")
    return mask


def apply_delta_with_token_scope(
    *,
    residual: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return residual + delta on masked positions; elsewhere residual unchanged.

    Shapes:
      residual: [B, T, D]
      delta: [B, T, D] or [B, D] (broadcast onto masked positions)
      mask: [B, T] bool
    """
    if delta.ndim == 2:
        delta = delta.unsqueeze(1).expand_as(residual)
    mask_expanded = mask.unsqueeze(-1)
    return torch.where(mask_expanded, residual + delta, residual)


def tag_hook_site(tensor: torch.Tensor, *, hook_site: str) -> torch.Tensor:
    """Attach a hook-site label for FEAT-036 alignment checks."""
    tensor.hook_site = hook_site  # type: ignore[attr-defined]
    return tensor


def read_hook_site(tensor: torch.Tensor) -> str | None:
    """Return the hook-site tag if present."""
    return getattr(tensor, "hook_site", None)



from dataclasses import dataclass


@dataclass(frozen=True)
class RealModelHookContractReport:
    """REAL-002 hook tensor contract diagnostics."""

    hook_module_name: str
    activation_shape: tuple[int, ...]
    activation_dtype: torch.dtype
    device: torch.device
    sequence_index_policy: str
    d_model: int
    decoder_width: int
    compatible: bool


def inspect_real_model_hook_contract(
    *,
    model_id: str,
    model_revision: str,
    prompts: Sequence[str],
    n_features: int,
    dtype: torch.dtype,
) -> RealModelHookContractReport:
    """Load a pinned causal LM and assert residual/hook tensor contracts."""
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision
    )
    model.eval()
    d_model = int(model.config.n_embd)
    hook_module_name = "transformer.h"
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    attention = encoded["attention_mask"]
    lengths = attention.sum(dim=1)
    # last_non_pad indexing
    residuals = torch.stack(
        [hidden[i, int(lengths[i].item()) - 1] for i in range(hidden.shape[0])],
        dim=0,
    ).to(dtype=dtype)
    decoder = torch.randn(n_features, d_model, dtype=dtype)
    compatible = decoder.shape[1] == residuals.shape[-1] == d_model
    return RealModelHookContractReport(
        hook_module_name=hook_module_name,
        activation_shape=tuple(int(x) for x in residuals.shape),
        activation_dtype=residuals.dtype,
        device=residuals.device,
        sequence_index_policy="last_non_pad",
        d_model=d_model,
        decoder_width=int(decoder.shape[1]),
        compatible=bool(compatible),
    )
