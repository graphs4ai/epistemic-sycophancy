"""Production margin projection batch (GRAD-004 / DEC-084): ∂M/∂x + JumpReLU latents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from epistemic_sycophancy.sae.jumprelu_delta import jumprelu
from epistemic_sycophancy.stack.resolver import resolve_resid_post_module


def compute_margin_projection_batch(
    stack: Any,
    *,
    layer: int,
    texts: Sequence[str],
    question_ids: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
) -> dict[str, Any]:
    """Return ∂M/∂x residual grads + JumpReLU latents at last prompt token.

    Unlike FS projection (∂φ/∂x), this backprops the semantic margin M so
    ``coefficient_jacobian`` yields ∂M/∂β (DEC-084).
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
    margins = []
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
        margins.append(diff)

    last_residual = torch.stack(last_rows, dim=0)
    margin_sum = torch.stack(margins).sum()
    grads = torch.autograd.grad(margin_sum, residual, retain_graph=False)[0]
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

    decoder = (
        sae.W_dec.detach()
        if hasattr(sae, "W_dec")
        else sae.decoder_weight.detach()
    )
    decoder_f64 = decoder.to(dtype=torch.float64)
    feature_scales = torch.linalg.vector_norm(decoder_f64, dim=1)

    return {
        "layer": layer,
        "residual_gradients": residual_gradients.to(dtype=torch.float64),
        "latents": latents.to(dtype=torch.float64),
        "decoder": decoder_f64,
        "feature_scales": feature_scales,
        "question_ids": [str(q) for q in question_ids],
    }
