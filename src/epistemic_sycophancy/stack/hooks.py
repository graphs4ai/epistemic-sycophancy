"""Multi-layer additive residual hooks (Phase K RUN-006)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from epistemic_sycophancy.intervention.hooks import (
    apply_delta_with_token_scope,
    build_token_scope_mask,
)
from epistemic_sycophancy.stack.resolver import resolve_resid_post_module


def _as_activation(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def _rebuild_output(output: Any, activation: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (activation,) + output[1:]
    return activation


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

    ``delta_fn`` is reserved for nonzero β (JumpReLU path in RUN-007+). For β=0
    this context manager short-circuits without calling ``delta_fn``.
    """
    del delta_fn  # unused while β=0 short-circuit is the only path
    beta_tensor = torch.as_tensor(list(beta), dtype=torch.float32)
    all_zero = bool(torch.all(beta_tensor == 0))

    layers_needed = sorted({layer for layer, _ in selected_keys})
    handles: list[Any] = []

    def make_hook(layer: int):
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            del module, inputs
            if all_zero:
                return output
            # Nonzero β path is implemented in later RUN cycles.
            raise NotImplementedError(
                "nonzero β multi-layer delta requires JumpReLU adapter (RUN-007)"
            )

        return hook

    for layer in layers_needed:
        if layer not in saes:
            raise KeyError(f"no SAE loaded for layer {layer}")
        module = resolve_resid_post_module(
            model, layer=layer, resolver_id=resolver_id
        )
        handles.append(module.register_forward_hook(make_hook(layer)))

    # Touch token-scope builders so DEC-015 stays wired for later nonzero β.
    _ = (token_scope, prompt_lengths, k, apply_delta_with_token_scope, build_token_scope_mask)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
