"""Multi-layer additive residual hooks (Phase K/L; DEC-053 / WIRE-001)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from epistemic_sycophancy.intervention.hooks import (
    apply_delta_with_token_scope,
    build_token_scope_mask,
)
from epistemic_sycophancy.sae.jumprelu_delta import apply_additive_jumprelu_sae_delta
from epistemic_sycophancy.stack.beta_layout import scatter_beta_by_layer
from epistemic_sycophancy.stack.resolver import resolve_resid_post_module


def _as_activation(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _rebuild_output(output: Any, activation: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (activation,) + output[1:]
    return activation


def _sae_tensor(sae: Any, *names: str) -> torch.Tensor:
    for name in names:
        if hasattr(sae, name):
            value = getattr(sae, name)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor):
                return value
        # SaeHandle wraps sae-lens SAE under .sae
        inner = getattr(sae, "sae", None)
        if inner is not None and hasattr(inner, name):
            value = getattr(inner, name)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor):
                return value
    raise AttributeError(
        f"SAE object missing tensor attributes {names!r}; got type {type(sae)!r}"
    )


def _default_jumprelu_delta(
    *,
    residual: torch.Tensor,
    selected_indices: Sequence[int],
    scales: Sequence[float],
    beta: Sequence[float],
    sae: Any,
) -> torch.Tensor:
    encoder_weight = _sae_tensor(sae, "encoder_weight", "W_enc")
    encoder_bias = _sae_tensor(sae, "encoder_bias", "b_enc")
    threshold = _sae_tensor(sae, "threshold", "threshold")
    decoder_weight = _sae_tensor(sae, "decoder_weight", "W_dec")
    # SaeHandle.decoder_weight may already be detached on the handle.
    if hasattr(sae, "decoder_weight") and isinstance(sae.decoder_weight, torch.Tensor):
        decoder_weight = sae.decoder_weight
    return apply_additive_jumprelu_sae_delta(
        residual=residual,
        selected_indices=selected_indices,
        scales=scales,
        beta=beta,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
    )


@contextmanager
def install_multi_layer_hooks(
    *,
    model: Any,
    resolver_id: str,
    saes: Mapping[int, Any],
    selected_keys: Sequence[tuple[int, int]],
    scales: Sequence[float],
    beta: Sequence[float],
    token_scope: str,
    prompt_lengths: Sequence[int],
    k: int | None = None,
    delta_fn: Any | None = None,
) -> Iterator[None]:
    """Install simultaneous resid_post hooks; at β=0 leave activations unchanged.

    Nonzero β applies JumpReLU additive delta (DEC-053) under ``token_scope``.
    """
    apply_delta = delta_fn or _default_jumprelu_delta
    beta_tensor = torch.as_tensor(list(beta), dtype=torch.float32)
    all_zero = bool(torch.all(beta_tensor == 0))
    by_layer = scatter_beta_by_layer(
        feature_ids=selected_keys,
        scales=scales,
        beta=beta,
    )

    layers_needed = sorted(by_layer.keys())
    handles: list[Any] = []

    def make_hook(layer: int):
        slice_ = by_layer[layer]
        sae = saes[layer]

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            del module, inputs
            if all_zero:
                return output
            activation = _as_activation(output)
            if activation.ndim != 3:
                raise ValueError(
                    f"expected resid_post activation [B,T,D]; got shape {tuple(activation.shape)}"
                )
            batch_size, seq_len, _d_model = activation.shape
            mask = build_token_scope_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                prompt_lengths=prompt_lengths,
                token_scope=token_scope,
                k=k,
                device=activation.device,
            )
            # Per-batch residual at the (first) masked position for last_prompt_token;
            # for multi-token scopes apply delta at each masked index.
            deltas = torch.zeros_like(activation)
            for batch_index in range(batch_size):
                positions = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
                for pos in positions.tolist():
                    token_residual = activation[batch_index, pos]
                    intervened = apply_delta(
                        residual=token_residual,
                        selected_indices=slice_.selected_indices,
                        scales=slice_.scales,
                        beta=slice_.beta,
                        sae=sae,
                    )
                    deltas[batch_index, pos] = intervened - token_residual
            # apply_delta_with_token_scope expects residual+delta on masked sites.
            # We already zeroed non-masked deltas, so mask keeps padding untouched.
            updated = apply_delta_with_token_scope(
                residual=activation,
                delta=deltas,
                mask=mask,
            )
            return _rebuild_output(output, updated)

        return hook

    for layer in layers_needed:
        if layer not in saes:
            raise KeyError(f"no SAE loaded for layer {layer}")
        module = resolve_resid_post_module(
            model, layer=layer, resolver_id=resolver_id
        )
        handles.append(module.register_forward_hook(make_hook(layer)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
