"""Feature-selection stage over multi-layer keys (Phase K RUN-010)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows


@dataclass(frozen=True)
class FeatureSelectionStageResult:
    """One-order feature-selection outputs keyed by (layer, feature_id)."""

    order_regime: str
    split_name: str
    signed_jacobians: dict[tuple[int, int], float]


def run_feature_selection_stage(
    *,
    order_regime: str,
    split_name: str,
    component_jacobians: Mapping[
        tuple[str, str], Mapping[tuple[int, int], float]
    ],
    freeze_status: str,
) -> FeatureSelectionStageResult:
    """Emit per-layer signed Jacobians for a single order on the FS split."""
    if split_name.startswith("holdout") or split_name == "holdout_test_behavior":
        load_holdout_rows(freeze_status=freeze_status)
        raise HoldoutAccessError(
            f"feature selection cannot use split {split_name!r}"
        )
    if split_name != "feature_selection":
        raise HoldoutAccessError(
            "feature selection stage requires split_name='feature_selection'; "
            f"got {split_name!r}"
        )

    signed: dict[tuple[int, int], float] = {}
    for (order, _component), scores in component_jacobians.items():
        if order != order_regime:
            continue
        for key, value in scores.items():
            if not (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], int)
                and isinstance(key[1], int)
            ):
                raise TypeError(
                    "Jacobian keys must be (layer, feature_id) tuples; "
                    f"got {key!r}"
                )
            # Keep max absolute contribution when components overlap (diagnostic merge).
            prev = signed.get(key)
            if prev is None or abs(value) > abs(prev):
                signed[key] = float(value)
    return FeatureSelectionStageResult(
        order_regime=order_regime,
        split_name=split_name,
        signed_jacobians=signed,
    )
