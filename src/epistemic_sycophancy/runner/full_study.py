"""full_study evaluation stage (ORCH-014 / DEC-069)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.evaluation.cross_order import build_cross_order_matrix
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.metrics.baseline_partition import (
    build_baseline_partition,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import compute_behavioral_metrics
from epistemic_sycophancy.optimization.checkpoint import load_checkpoint


def run_full_study_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    eval_payload: Mapping[str, Any],
    holdout_question_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate best β on behavior_validation; write metrics + 3×3; no holdout."""
    if freeze_status != "sealed":
        raise HoldoutAccessError(
            "full_study requires freeze_status='sealed' "
            f"(got {freeze_status!r}; DEC-055 / DEC-069)"
        )
    if holdout_question_ids:
        # Ignore holdout IDs; never call the holdout loader from full_study.
        del holdout_question_ids

    ckpt_path = Path(study.run.artifact_dir) / "optimize" / "best_checkpoint.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing best checkpoint at {ckpt_path}")
    ckpt = load_checkpoint(json.loads(ckpt_path.read_text(encoding="utf-8")))
    beta = tuple(float(x) for x in ckpt["beta"])

    neutral = dict(eval_payload["current_neutral_margins"])
    ib = dict(eval_payload["current_ib_margins"])
    cb = dict(eval_payload["current_cb_margins"])
    baselines = dict(eval_payload["baseline_neutral_margins_by_order"])
    epsilon = float(study.experiment.tie_band_epsilon)
    tie_policy = str(study.experiment.tie_policy)

    # Primary CF partition for behavior_validation metrics.
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=baselines.get("CF", neutral),
        epsilon=epsilon,
        tie_policy=tie_policy,
    )
    artifact = freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash=study.stack.model.revision,
        prompt_template_hash="orch-full-study-prompt",
        order_manifest_hash="orch-full-study-order-CF",
        dataset_manifest_hash="orch-full-study-dataset",
    )
    metrics = compute_behavioral_metrics(
        frozen_partition=artifact,
        current_neutral_margins=neutral,
        current_ib_margins=ib,
        current_cb_margins=cb,
        epsilon=epsilon,
    )

    metrics_by_order: dict[str, dict[str, float | int]] = {}
    fingerprints: dict[str, str] = {}
    for order in ("CF", "IF", "RO"):
        part = build_baseline_partition(
            order_regime=order,
            neutral_margins=baselines.get(order, neutral),
            epsilon=epsilon,
            tie_policy=tie_policy,
        )
        art = freeze_baseline_partition_artifact(
            partition=part,
            model_revision_hash=study.stack.model.revision,
            prompt_template_hash="orch-full-study-prompt",
            order_manifest_hash=f"orch-full-study-order-{order}",
            dataset_manifest_hash="orch-full-study-dataset",
        )
        fingerprints[order] = art.fingerprint
        m = compute_behavioral_metrics(
            frozen_partition=art,
            current_neutral_margins=neutral,
            current_ib_margins=ib,
            current_cb_margins=cb,
            epsilon=epsilon,
        )
        metrics_by_order[order] = {
            "ftw": float(m.ftw if m.ftw is not None else 0.0),
            "cbr": float(m.cbr if m.cbr is not None else 0.0),
            "selectivity": float(m.selectivity if m.selectivity is not None else 0.0),
            "n_q_plus": int(m.n_q_plus),
            "n_q_minus": int(m.n_q_minus),
        }

    cells = build_cross_order_matrix(
        betas_by_optimized_under={"CF": beta, "IF": beta, "RO": beta},
        optimization_order_manifest_hashes={
            "CF": "opt-CF",
            "IF": "opt-IF",
            "RO": "opt-RO",
        },
        evaluation_order_manifest_hashes={
            "CF": "eval-CF",
            "IF": "eval-IF",
            "RO": "eval-RO",
        },
        baseline_partition_fingerprints=fingerprints,
        metrics_by_evaluated_under=metrics_by_order,
    )

    out_dir = Path(study.run.artifact_dir) / "full_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    behavioral_path = out_dir / "behavioral.json"
    matrix_path = out_dir / "cross_order_matrix.json"
    behavioral_payload = {
        "split": "behavior_validation",
        "beta": list(beta),
        "neutral_accuracy": metrics.neutral_accuracy,
        "ftw": metrics.ftw,
        "cbr": metrics.cbr,
        "selectivity": metrics.selectivity,
        "pra_mean": metrics.pra_mean,
        "pra_all": metrics.pra_all,
        "n_q_plus": metrics.n_q_plus,
        "n_q_minus": metrics.n_q_minus,
        "validation_question_ids": list(eval_payload.get("validation_question_ids", ())),
    }
    behavioral_path.write_text(
        json.dumps(behavioral_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    matrix_payload = {
        "cells": [asdict(cell) for cell in cells],
    }
    matrix_path.write_text(
        json.dumps(matrix_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": {
            "ftw": metrics.ftw,
            "cbr": metrics.cbr,
            "selectivity": metrics.selectivity,
            "n_cells": len(cells),
            "holdout_accessed": False,
        },
        "artifacts": {
            "behavioral": str(behavioral_path),
            "cross_order_matrix": str(matrix_path),
        },
    }
