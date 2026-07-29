"""Feature scales from SAE decoder norms (Phase L WIRE-008 / DEC-061)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def decoder_row_norms(
    decoder_weight: torch.Tensor,
    feature_ids: Sequence[int],
) -> tuple[float, ...]:
    """Return L2 norms of decoder rows for ``feature_ids`` (scale_source=decoder_norm)."""
    if decoder_weight.ndim != 2:
        raise ValueError(
            f"decoder_weight must be [n_features, d_model]; got {tuple(decoder_weight.shape)}"
        )
    norms: list[float] = []
    for feature_id in feature_ids:
        row = decoder_weight[int(feature_id)].detach().float()
        norm = float(torch.linalg.vector_norm(row).item())
        if not (norm > 0.0):
            raise ValueError(
                f"decoder_norm for feature_id={feature_id} must be > 0; got {norm}"
            )
        norms.append(norm)
    return tuple(norms)


def scales_for_layer_feature_keys(
    *,
    keys: Sequence[tuple[int, int]],
    saes: Mapping[int, Any],
    scale_source: str,
) -> tuple[float, ...]:
    """Build aligned scales for (layer, feature_id) keys (DEC-061)."""
    if scale_source != "decoder_norm":
        raise ValueError(
            f"unsupported scale_source {scale_source!r}; DEC-061 requires 'decoder_norm'"
        )
    scales: list[float] = []
    for layer, feature_id in keys:
        if layer not in saes:
            raise KeyError(f"no SAE for layer {layer}")
        handle = saes[layer]
        if hasattr(handle, "decoder_weight"):
            decoder = handle.decoder_weight
        else:
            decoder = handle.W_dec
        scales.append(decoder_row_norms(decoder, (feature_id,))[0])
    return tuple(scales)
