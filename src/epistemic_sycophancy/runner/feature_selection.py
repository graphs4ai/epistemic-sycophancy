"""Feature-selection stage over multi-layer keys (Phase K/L RUN-010 / WIRE-007)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows
from epistemic_sycophancy.reproducibility.phase_gates import (
    require_feature_selection_split_gate,
)


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
            prev = signed.get(key)
            if prev is None or abs(value) > abs(prev):
                signed[key] = float(value)
    return FeatureSelectionStageResult(
        order_regime=order_regime,
        split_name=split_name,
        signed_jacobians=signed,
    )


def run_feature_selection_stage_computed(
    *,
    order_regime: str,
    split_name: str,
    question_ids: Sequence[str],
    jacobian_fn: Callable[..., Mapping[tuple[int, int], float]],
    freeze_status: str,
    optimization_question_ids: Sequence[str],
    validation_question_ids: Sequence[str],
    holdout_question_ids: Sequence[str],
) -> FeatureSelectionStageResult:
    """Compute projected Jacobians on a tiny FS subset (DEC-060 ranking authority)."""
    del freeze_status
    if split_name != "feature_selection":
        raise HoldoutAccessError(
            "feature selection stage requires split_name='feature_selection'; "
            f"got {split_name!r}"
        )
    qids = tuple(str(q) for q in question_ids)
    require_feature_selection_split_gate(
        artifact_question_ids=qids,
        feature_selection_question_ids=qids,
        optimization_question_ids=optimization_question_ids,
        validation_question_ids=validation_question_ids,
        holdout_question_ids=holdout_question_ids,
    )
    downstream = (
        set(str(q) for q in optimization_question_ids)
        | set(str(q) for q in validation_question_ids)
        | set(str(q) for q in holdout_question_ids)
    )
    overlap = set(qids) & downstream
    if overlap:
        raise HoldoutAccessError(
            f"FS smoke questions leak into opt/val/holdout: {sorted(overlap)}"
        )
    signed = dict(jacobian_fn(order_regime=order_regime, question_ids=qids))
    for key in signed:
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
    return FeatureSelectionStageResult(
        order_regime=order_regime,
        split_name=split_name,
        signed_jacobians={k: float(v) for k, v in signed.items()},
    )
