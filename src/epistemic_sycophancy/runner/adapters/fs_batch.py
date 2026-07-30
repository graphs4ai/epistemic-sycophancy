"""Production FS projection batch for InterventionStack (ORCH-036 / DEC-060)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from epistemic_sycophancy.feature_selection.components import (
    COMPONENT_CONDITION,
    logistic_preservation_surrogate,
)
from epistemic_sycophancy.feature_selection.projected_gradient import (
    question_macro_prompt_weights,
)
from epistemic_sycophancy.sae.jumprelu_delta import jumprelu
from epistemic_sycophancy.stack.resolver import resolve_resid_post_module


def fs_component_margin_loss(
    *,
    margin: torch.Tensor,
    tau: float,
    component: str,
) -> torch.Tensor:
    """Per-prompt FS component loss: φ(M)=softplus(-M/τ) for all four (DEC-085).

    Preservation components must use this logistic surrogate, never the
    baseline-relative hinges (FEAT-011/012 / FSC-004).
    """
    if component not in COMPONENT_CONDITION:
        raise ValueError(f"unknown FS component {component!r}")
    return logistic_preservation_surrogate(margin=margin, tau=tau)


def weighted_component_residual_grads(
    *,
    prompt_losses: torch.Tensor,
    residual: torch.Tensor,
    question_ids: Sequence[object],
) -> torch.Tensor:
    """Backward §11.3 weighted scalar once: Σ_p w_p φ_p with w_p=1/(|Q|·|B_q|).

    Returns ∂(weighted loss)/∂residual. Callers must not re-apply prompt weights
    after projection (FSC-003 / FEAT-014); use ``sum_coefficient_jacobians``.
    """
    if prompt_losses.ndim != 1:
        raise ValueError(
            f"prompt_losses must be rank-1 [batch]; got {tuple(prompt_losses.shape)}"
        )
    if prompt_losses.shape[0] != len(question_ids):
        raise ValueError("prompt_losses length must match question_ids")
    weights = question_macro_prompt_weights(question_ids=question_ids).to(
        dtype=prompt_losses.dtype, device=prompt_losses.device
    )
    weighted = (weights * prompt_losses).sum()
    grads = torch.autograd.grad(weighted, residual, retain_graph=False)[0]
    return grads


def compute_fs_projection_batch(
    stack: Any,
    *,
    layer: int,
    texts: Sequence[str],
    question_ids: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
    tau: float,
    use_question_macro_weights: bool = True,
    prompt_batch_size: int | None = None,
) -> dict[str, Any]:
    """Return residual grads + JumpReLU latents for one layer (last prompt token).

    Loss is logistic margin loss φ(M)=softplus(-M/τ). When
    ``use_question_macro_weights`` is True (default, DEC-085 / FSC-003), the
    scalar is Σ_p w_p φ_p with w_p=1/(|Q|·|B_q|) so downstream aggregation must
    **sum** per-prompt Jacobians (not apply question-macro again).

    ``prompt_batch_size`` microbatches the forward/backward to bound VRAM
    (ASAP smoke with many IB/CB variants).
    """
    if len(texts) != len(question_ids) or len(texts) != len(truthful_labels):
        raise ValueError("texts, question_ids, and truthful_labels must align")
    if len(continuation_token_ids_A) != 1 or len(continuation_token_ids_B) != 1:
        raise ValueError("single-token A/B continuations required")

    n = len(texts)
    chunk = int(prompt_batch_size) if prompt_batch_size is not None else n
    if chunk <= 0:
        raise ValueError(f"prompt_batch_size must be positive; got {prompt_batch_size!r}")

    # Global question-macro weights over the full component batch (FEAT-014).
    weights = question_macro_prompt_weights(question_ids=question_ids)

    all_grads: list[torch.Tensor] = []
    all_latents: list[torch.Tensor] = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        micro = _fs_projection_microbatch(
            stack,
            layer=layer,
            texts=texts[start:end],
            question_ids=question_ids[start:end],
            continuation_token_ids_A=continuation_token_ids_A,
            continuation_token_ids_B=continuation_token_ids_B,
            truthful_labels=truthful_labels[start:end],
            tau=tau,
            prompt_weights=(
                weights[start:end]
                if use_question_macro_weights
                else torch.full((end - start,), 1.0 / n, dtype=torch.float64)
            ),
        )
        all_grads.append(micro["residual_gradients"])
        all_latents.append(micro["latents"])

    return {
        "layer": layer,
        "residual_gradients": torch.cat(all_grads, dim=0),
        "latents": torch.cat(all_latents, dim=0),
        "question_ids": [str(q) for q in question_ids],
        "weights_applied": True,
    }


def _fs_projection_microbatch(
    stack: Any,
    *,
    layer: int,
    texts: Sequence[str],
    question_ids: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
    tau: float,
    prompt_weights: torch.Tensor,
) -> dict[str, Any]:
    """One microbatch forward/backward with precomputed prompt weights."""
    device = stack.device
    tokenizer = stack.tokenizer
    model = stack.model
    encoded = tokenizer(list(texts), return_tensors="pt", padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    attention = encoded.get("attention_mask")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        leaf = tensor.detach().requires_grad_(True)
        captured["residual"] = leaf
        if isinstance(output, tuple):
            return (leaf,) + tuple(output[1:])
        return leaf

    module = resolve_resid_post_module(
        model,
        layer=layer,
        resolver_id=stack.config.hooks.resolver_id,
    )
    handle = module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=attention,
            )
            logits = outputs.logits
    finally:
        handle.remove()

    residual = captured["residual"]
    batch = residual.shape[0]
    last_rows = []
    losses = []
    token_a = int(continuation_token_ids_A[0])
    token_b = int(continuation_token_ids_B[0])
    for i in range(batch):
        if attention is not None:
            prompt_length = int(attention[i].sum().item())
        else:
            prompt_length = int(encoded["input_ids"].shape[1])
        last_rows.append(residual[i, prompt_length - 1, :])
        last_logit = logits[i, prompt_length - 1, :]
        diff = last_logit[token_a] - last_logit[token_b]
        if str(truthful_labels[i]).upper() == "B":
            diff = -diff
        losses.append(torch.nn.functional.softplus(-diff / float(tau)))

    last_residual = torch.stack(last_rows, dim=0)
    prompt_losses = torch.stack(losses)
    w = prompt_weights.to(dtype=prompt_losses.dtype, device=prompt_losses.device)
    weighted = (w * prompt_losses).sum()
    grads = torch.autograd.grad(weighted, residual, retain_graph=False)[0]
    residual_grads = []
    for i in range(batch):
        if attention is not None:
            prompt_length = int(attention[i].sum().item())
        else:
            prompt_length = int(encoded["input_ids"].shape[1])
        residual_grads.append(grads[i, prompt_length - 1, :].detach().float())
    residual_gradients = torch.stack(residual_grads, dim=0)

    sae_handle = stack.saes[layer]
    sae = sae_handle.sae
    enc_w = sae.W_enc.detach()
    enc_b = sae.b_enc.detach()
    threshold = sae.threshold.detach() if hasattr(sae, "threshold") else None
    if threshold is None:
        threshold = getattr(sae, "b_mag", None)
    if threshold is None:
        raise ValueError(f"SAE at layer {layer} missing JumpReLU threshold")
    work = last_residual.detach().float()
    pre = work @ enc_w.float() + enc_b.float()
    latents = jumprelu(pre, threshold.float())

    del residual, grads, outputs, logits, encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "residual_gradients": residual_gradients.to(dtype=torch.float64),
        "latents": latents.to(dtype=torch.float64),
        "question_ids": [str(q) for q in question_ids],
    }
