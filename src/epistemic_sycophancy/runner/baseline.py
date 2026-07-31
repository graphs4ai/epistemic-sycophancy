"""Baseline partition stage orchestration (ORCH-003 / DEC-070 / DEC-087)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig, study_order_regime
from epistemic_sycophancy.logging.pipeline import log_progress
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
    order_regime: str | None = None,
) -> dict[str, Any]:
    """Score IDs, build one FS partition for the study order, write artifact."""
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
    regime = (
        str(order_regime).upper()
        if order_regime is not None
        else study_order_regime(study)
    )
    fingerprint = study_config_fingerprint(study)
    out_dir = Path(study.run.artifact_dir) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    partition = run_baseline_partition_stage_via_scores(
        split_name=resolved_split,
        order_regime=regime,
        question_ids=qids,
        score_fn=score_fn,
        epsilon=float(study.experiment.tie_band_epsilon),
        tie_policy=str(study.experiment.tie_policy),
        freeze_status=freeze_status,
    )
    artifact = freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash=study.stack.model.revision,
        prompt_template_hash="orch-baseline-prompt",
        order_manifest_hash=f"orch-order-{regime}",
        dataset_manifest_hash="orch-dataset",
    )
    path = out_dir / f"partition_{regime}.json"
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
    log_progress(
        "baseline_partition",
        order_regime=regime,
        split_name=resolved_split,
        n_q_plus=len(partition.q_plus),
        n_q_minus=len(partition.q_minus),
        n_q_tie=partition.n_q_tie,
        path=str(path),
    )
    artifacts = {
        f"partition_{regime}": str(path),
        "partition": str(path),
    }
    metrics: dict[str, Any] = {
        "order_regime": partition.order_regime,
        "order_regimes": [regime],
        "n_q_plus": len(partition.q_plus),
        "n_q_minus": len(partition.q_minus),
        "n_q_tie": partition.n_q_tie,
        f"n_q_plus_{regime}": len(partition.q_plus),
        f"n_q_minus_{regime}": len(partition.q_minus),
        "q_plus": sorted(partition.q_plus),
        "q_minus": sorted(partition.q_minus),
    }
    return {
        "partition": partition,
        "metrics": metrics,
        "artifacts": artifacts,
    }
