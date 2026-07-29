"""Staged runner steps for baseline partitions (Phase K/L RUN-009 / WIRE-006)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.metrics.baseline_partition import (
    BaselinePartition,
    build_baseline_partition,
)
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows

_ALLOWED_BASELINE_SPLITS = frozenset({"feature_selection"})


def run_baseline_partition_stage(
    *,
    split_name: str,
    order_regime: str,
    neutral_margins: Mapping[str, float],
    epsilon: float,
    tie_policy: str,
    freeze_status: str,
) -> BaselinePartition:
    """Build baseline partition from FS-split neutrals; holdout stays sealed."""
    if split_name == "holdout_test_behavior" or split_name.startswith("holdout"):
        load_holdout_rows(freeze_status=freeze_status)
        raise HoldoutAccessError(
            f"baseline partition stage cannot use split {split_name!r}"
        )
    if split_name not in _ALLOWED_BASELINE_SPLITS:
        raise HoldoutAccessError(
            "baseline partition stage allows only feature_selection; "
            f"got {split_name!r}"
        )
    return build_baseline_partition(
        order_regime=order_regime,
        neutral_margins=neutral_margins,
        epsilon=epsilon,
        tie_policy=tie_policy,
    )


def run_baseline_partition_stage_via_scores(
    *,
    split_name: str,
    order_regime: str,
    question_ids: Sequence[str],
    score_fn: Callable[[Sequence[str]], Mapping[str, float]],
    epsilon: float,
    tie_policy: str,
    freeze_status: str,
) -> BaselinePartition:
    """Score FS subset via ``score_fn`` then build frozen baseline partition."""
    if split_name == "holdout_test_behavior" or split_name.startswith("holdout"):
        load_holdout_rows(freeze_status=freeze_status)
        raise HoldoutAccessError(
            f"baseline partition stage cannot use split {split_name!r}"
        )
    if split_name not in _ALLOWED_BASELINE_SPLITS:
        raise HoldoutAccessError(
            "baseline partition stage allows only feature_selection; "
            f"got {split_name!r}"
        )
    neutral_margins = dict(score_fn(tuple(question_ids)))
    return build_baseline_partition(
        order_regime=order_regime,
        neutral_margins=neutral_margins,
        epsilon=epsilon,
        tie_policy=tie_policy,
    )
