"""Resolve Gemma-3 resid_post hook modules by layer (Phase K RUN-005)."""

from __future__ import annotations

from typing import Any


class UnknownHookLayerError(Exception):
    """Raised when a requested layer index is outside the model stack."""


def resolve_resid_post_module(
    model: Any,
    *,
    layer: int,
    resolver_id: str,
) -> Any:
    """Return the module whose output is the residual stream at ``resid_post``.

    For ``resolver_id=\"gemma3_resid_post\"`` (DEC-049 probe): path is
    ``model.language_model.layers[layer]`` on ``Gemma3ForConditionalGeneration``.
    Hooking that decoder layer's forward output is the post-layer residual.
    """
    if resolver_id != "gemma3_resid_post":
        raise ValueError(f"unsupported resolver_id: {resolver_id!r}")
    try:
        layers = model.model.language_model.layers
    except AttributeError as exc:
        raise ValueError(
            "gemma3_resid_post expects model.model.language_model.layers"
        ) from exc
    n_layers = len(layers)
    if layer < 0 or layer >= n_layers:
        raise UnknownHookLayerError(
            f"layer {layer} out of range for n_layers={n_layers}"
        )
    return layers[layer]
