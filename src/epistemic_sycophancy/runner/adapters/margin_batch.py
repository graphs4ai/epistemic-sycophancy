"""Production margin projection batch (GRAD-004 / DEC-084): ∂M/∂x + JumpReLU latents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from epistemic_sycophancy.runner.progress import tick_prompt_batch
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
    prompt_batch_size: int | None = None,
) -> dict[str, Any]:
    """Return ∂M/∂x residual grads + JumpReLU latents at last prompt token.

    Unlike FS projection (∂φ/∂x), this backprops the semantic margin M so
    ``coefficient_jacobian`` yields ∂M/∂β (DEC-084).

    ``prompt_batch_size`` microbatches the forward/backward to bound VRAM
    (DEC-090; parity with FS DEC-085 / DEC-022). ``None`` → one full batch.
    """
    if len(texts) != len(question_ids) or len(texts) != len(truthful_labels):
        raise ValueError("texts, question_ids, and truthful_labels must align")
    if len(continuation_token_ids_A) != 1 or len(continuation_token_ids_B) != 1:
        raise ValueError("single-token A/B continuations required")

    n = len(texts)
    chunk = int(prompt_batch_size) if prompt_batch_size is not None else n
    if chunk <= 0:
        raise ValueError(f"prompt_batch_size must be positive; got {prompt_batch_size!r}")

    all_grads: list[torch.Tensor] = []
    all_latents: list[torch.Tensor] = []
    decoder_f64: torch.Tensor | None = None
    feature_scales: torch.Tensor | None = None
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        micro = _margin_projection_microbatch(
            stack,
            layer=layer,
            texts=texts[start:end],
            question_ids=question_ids[start:end],
            continuation_token_ids_A=continuation_token_ids_A,
            continuation_token_ids_B=continuation_token_ids_B,
            truthful_labels=truthful_labels[start:end],
        )
        all_grads.append(micro["residual_gradients"])
        all_latents.append(micro["latents"])
        if decoder_f64 is None:
            decoder_f64 = micro["decoder"]
            feature_scales = micro["feature_scales"]
        tick_prompt_batch()

    assert decoder_f64 is not None and feature_scales is not None
    return {
        "layer": layer,
        "residual_gradients": torch.cat(all_grads, dim=0),
        "latents": torch.cat(all_latents, dim=0),
        "decoder": decoder_f64,
        "feature_scales": feature_scales,
        "question_ids": [str(q) for q in question_ids],
    }


def _margin_projection_microbatch(
    stack: Any,
    *,
    layer: int,
    texts: Sequence[str],
    question_ids: Sequence[str],
    continuation_token_ids_A: Sequence[int],
    continuation_token_ids_B: Sequence[int],
    truthful_labels: Sequence[str],
) -> dict[str, Any]:
    """One microbatch forward/backward for ∂M/∂x + JumpReLU latents."""
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

    del residual, grads, outputs, logits, encoded

    return {
        "residual_gradients": residual_gradients.to(dtype=torch.float64),
        "latents": latents.to(dtype=torch.float64),
        "decoder": decoder_f64,
        "feature_scales": feature_scales,
        "question_ids": [str(q) for q in question_ids],
    }
