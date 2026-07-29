"""Production scale_fn adapter (ORCH-022 / DEC-061)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.stack.scales import scales_for_layer_feature_keys


def build_scale_fn(
    study: StudyConfig,
    stack: Any,
) -> Callable[[Sequence[tuple[int, int]]], Mapping[tuple[int, int], float]]:
    """Build ``keys -> scales`` using decoder_norm (DEC-061)."""
    del study  # scale_source is fixed by DEC-061

    def scale_fn(
        keys: Sequence[tuple[int, int]],
    ) -> Mapping[tuple[int, int], float]:
        scales = scales_for_layer_feature_keys(
            keys=keys,
            saes=stack.saes,
            scale_source="decoder_norm",
        )
        return {key: float(scale) for key, scale in zip(keys, scales, strict=True)}

    return scale_fn
