"""Production FS projection batch for InterventionStack (ORCH-036 / DEC-060)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from epistemic_sycophancy.feature_selection.projected_gradient import (
    question_macro_prompt_weights,
)
from epistemic_sycophancy.sae.jumprelu_delta import jumprelu
from epistemic_sycophancy.stack.resolver import resolve_resid_post_module


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
) -> dict[str, Any]:
    """Return residual grads + JumpReLU latents for one layer (last prompt token).

    Loss is logistic margin loss φ(M)=softplus(-M/τ). When
    ``use_question_macro_weights`` is True (default, DEC-085 / FSC-003), the
    scalar is Σ_p w_p φ_p with w_p=1/(|Q|·|B_q|) so downstream aggregation must
    **sum** per-prompt Jacobians (not apply question-macro again).
    """
    if len(texts) != len(question_ids) or len(texts) != len(truthful_labels):
        raise ValueError("texts, question_ids, and truthful_labels must align")
    if len(continuation_token_ids_A) != 1 or len(continuation_token_ids_B) != 1:
        raise ValueError("single-token A/B continuations required")

    device = stack.device
    tokenizer = stack.tokenizer
    model = stack.model
    encoded = tokenizer(list(texts), return_tensors="pt", padding=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    attention = encoded.get("attention_mask")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        # Detach upstream (frozen weights) and re-enable local grads for ∂ℓ/∂x.
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
            logits = outputs.logits  # [B, T, V]
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
    if use_question_macro_weights:
        grads = weighted_component_residual_grads(
            prompt_losses=prompt_losses,
            residual=residual,
            question_ids=question_ids,
        )
    else:
        loss = prompt_losses.mean()
        grads = torch.autograd.grad(loss, residual, retain_graph=False)[0]
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
        # Some JumpReLU SAEs store threshold on cfg / attribute aliases.
        threshold = getattr(sae, "b_mag", None)
    if threshold is None:
        raise ValueError(f"SAE at layer {layer} missing JumpReLU threshold")
    work = last_residual.detach().float()
    # GemmaScope2 sae-lens W_enc is [d_in, n_features] (transposed vs Phase E).
    pre = work @ enc_w.float() + enc_b.float()
    latents = jumprelu(pre, threshold.float())

    return {
        "layer": layer,
        "residual_gradients": residual_gradients.to(dtype=torch.float64),
        "latents": latents.to(dtype=torch.float64),
        "question_ids": [str(q) for q in question_ids],
        "weights_applied": bool(use_question_macro_weights),
    }
