"""Baseline partition stage orchestration (ORCH-003 / DEC-070)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.metrics.baseline_partition import (
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.runner.stages import run_baseline_partition_stage_via_scores


def run_baseline_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    score_fn: Callable[[Sequence[str]], Mapping[str, float]],
    question_ids: Sequence[str] | None = None,
    split_name: str | None = None,
    order_regimes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score smoke (or override) IDs, build FS partitions for each order, write artifacts."""
    smoke = study.run.smoke
    if question_ids is None:
        if smoke.question_ids is not None:
            qids = tuple(smoke.question_ids)
        else:
            raise ValueError(
                "baseline dispatch requires question_ids when smoke uses n_questions "
                "path without corpus injection"
            )
    else:
        qids = tuple(str(q) for q in question_ids)

    resolved_split = split_name if split_name is not None else "feature_selection"
    regimes = (
        tuple(str(o) for o in order_regimes)
        if order_regimes is not None
        else tuple(str(o) for o in study.run.order_regimes) or ("CF",)
    )
    fingerprint = study_config_fingerprint(study)
    out_dir = Path(study.run.artifact_dir) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    metrics: dict[str, Any] = {"order_regimes": list(regimes)}
    last_partition = None
    for order_regime in regimes:
        # Per-order scoring: score_fn may close over a fixed order; callers that
        # need multi-order must pass an order-aware score_fn or invoke once per order.
        partition = run_baseline_partition_stage_via_scores(
            split_name=resolved_split,
            order_regime=order_regime,
            question_ids=qids,
            score_fn=score_fn,
            epsilon=float(study.experiment.tie_band_epsilon),
            tie_policy=str(study.experiment.tie_policy),
            freeze_status=freeze_status,
        )
        last_partition = partition
        artifact = freeze_baseline_partition_artifact(
            partition=partition,
            model_revision_hash=study.stack.model.revision,
            prompt_template_hash="orch-baseline-prompt",
            order_manifest_hash=f"orch-order-{order_regime}",
            dataset_manifest_hash="orch-dataset",
        )
        path = out_dir / f"partition_{order_regime}.json"
        payload = {
            "order_regime": artifact.order_regime,
            "q_plus": sorted(artifact.partition.q_plus),
            "q_minus": sorted(artifact.partition.q_minus),
            "q_tie": sorted(artifact.partition.q_tie),
            "n_q_plus": len(artifact.partition.q_plus),
            "n_q_minus": len(artifact.partition.q_minus),
            "n_q_tie": artifact.partition.n_q_tie,
            "fingerprint": artifact.fingerprint,
            "study_yaml_fingerprint": fingerprint,
            "question_ids": list(qids),
            "split_name": resolved_split,
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        artifacts[f"partition_{order_regime}"] = str(path)
        metrics[f"n_q_plus_{order_regime}"] = len(partition.q_plus)
        metrics[f"n_q_minus_{order_regime}"] = len(partition.q_minus)

    assert last_partition is not None
    metrics.update(
        {
            "n_q_plus": len(last_partition.q_plus),
            "n_q_minus": len(last_partition.q_minus),
            "n_q_tie": last_partition.n_q_tie,
            "order_regime": last_partition.order_regime,
            "q_plus": sorted(last_partition.q_plus),
            "q_minus": sorted(last_partition.q_minus),
        }
    )
    # Back-compat single key used by ORCH-003 tests.
    if "partition_CF" in artifacts:
        artifacts["partition"] = artifacts["partition_CF"]
    elif artifacts:
        artifacts["partition"] = next(iter(artifacts.values()))

    return {
        "partition": last_partition,
        "metrics": metrics,
        "artifacts": artifacts,
    }
