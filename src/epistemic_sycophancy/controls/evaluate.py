"""Shared control/primary evaluation helper (Phase I CTRL-006)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InterventionEvalResult:
    """Result of evaluating an intervention through the shared pipeline."""

    pipeline_id: str
    prompts: tuple[Any, ...]
    feature_ids: tuple[int, ...]
    betas: tuple[float, ...]
    metrics: Mapping[str, float]
    metric_fn: Callable[..., Mapping[str, float]]


def evaluate_intervention_metrics(
    *,
    prompts: Sequence[Any],
    feature_ids: Sequence[int],
    betas: Sequence[float],
    metric_fn: Callable[..., Mapping[str, float]],
    pipeline_id: str,
) -> InterventionEvalResult:
    """Evaluate primary or control via the same prompts/metric path (CTRL-006)."""
    prompt_tuple = tuple(prompts)
    feature_tuple = tuple(int(x) for x in feature_ids)
    beta_tuple = tuple(float(x) for x in betas)
    metrics = metric_fn(
        feature_ids=feature_tuple,
        betas=beta_tuple,
        prompts=prompt_tuple,
    )
    return InterventionEvalResult(
        pipeline_id=pipeline_id,
        prompts=prompt_tuple,
        feature_ids=feature_tuple,
        betas=beta_tuple,
        metrics=dict(metrics),
        metric_fn=metric_fn,
    )
