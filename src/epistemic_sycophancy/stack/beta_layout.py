"""Align CFG β with CommonFeaturePool and scatter by layer (DEC-054 / RUN-011)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.pool import CommonFeaturePool


@dataclass(frozen=True)
class LayerBetaSlice:
    """Per-layer selected indices with aligned scales and β (DEC-018/054)."""

    selected_indices: tuple[int, ...]
    scales: tuple[float, ...]
    beta: tuple[float, ...]


def align_cfg_feature_ids_with_pool(
    pool: CommonFeaturePool,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Return CFG-ready feature_ids and scales matching the common pool order."""
    return pool.feature_ids, pool.scales


def scatter_beta_by_layer(
    *,
    feature_ids: Sequence[tuple[int, int]],
    scales: Sequence[float],
    beta: Sequence[float],
) -> dict[int, LayerBetaSlice]:
    """Group selected (layer, feature_id) entries into per-layer SAE index slices."""
    if not (len(feature_ids) == len(scales) == len(beta)):
        raise ValueError(
            "feature_ids, scales, and beta must have equal length; "
            f"got {len(feature_ids)}, {len(scales)}, {len(beta)}"
        )
    buckets: dict[int, list[tuple[int, float, float]]] = {}
    for (layer, feature_id), scale, coef in zip(feature_ids, scales, beta):
        buckets.setdefault(layer, []).append((int(feature_id), float(scale), float(coef)))
    result: dict[int, LayerBetaSlice] = {}
    for layer, rows in buckets.items():
        rows_sorted = sorted(rows, key=lambda row: row[0])
        result[layer] = LayerBetaSlice(
            selected_indices=tuple(r[0] for r in rows_sorted),
            scales=tuple(r[1] for r in rows_sorted),
            beta=tuple(r[2] for r in rows_sorted),
        )
    return result
