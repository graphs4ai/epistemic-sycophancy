"""Control evaluation pipeline parity tests (Phase I CTRL-006)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.controls.evaluate import evaluate_intervention_metrics


@pytest.mark.unit
def test_controls__evaluation_pipeline__uses_same_prompts_scoring_and_metrics_as_primary_intervention() -> None:
    """CTRL-006: controls use the same prompts/scoring/metrics path as primary.

    Only feature IDs / β assignment differ; the metric function and prompt set
    identity must be shared.
    """
    prompts = ("p1", "p2")
    primary = evaluate_intervention_metrics(
        prompts=prompts,
        feature_ids=[1, 2],
        betas=[-1.0, -0.5],
        metric_fn=_stub_metric,
        pipeline_id="shared_eval_v1",
    )
    control = evaluate_intervention_metrics(
        prompts=prompts,
        feature_ids=[3, 4],
        betas=[-0.5, -1.0],
        metric_fn=_stub_metric,
        pipeline_id="shared_eval_v1",
    )
    assert primary.pipeline_id == control.pipeline_id == "shared_eval_v1"
    assert primary.prompts is control.prompts or primary.prompts == control.prompts
    assert primary.metric_fn is control.metric_fn
    # Intervention differs; shared path still produces metrics dicts.
    assert set(primary.metrics) == set(control.metrics) == {"selectivity"}


def _stub_metric(*, feature_ids, betas, prompts):  # noqa: ANN001
    del prompts
    return {"selectivity": float(sum(betas)) - float(sum(feature_ids))}
