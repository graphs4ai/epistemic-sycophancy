"""full_study evaluation stage (ORCH-014 / DEC-069 / DEC-087 / DEC-098 / DEC-100)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig, study_order_regime
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.logging.pipeline import log_progress
from epistemic_sycophancy.metrics.baseline_partition import (
    build_baseline_partition,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import (
    BehavioralMetrics,
    compute_behavioral_metrics,
)
from epistemic_sycophancy.optimization.checkpoint import load_checkpoint

_LOSS_CRITERIA: tuple[str, ...] = (
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
)
_BEST_BY_RE = re.compile(
    r"^best_checkpoint_by_(" + "|".join(_LOSS_CRITERIA) + r")\.json$"
)


def _behavioral_payload(
    *,
    metrics: BehavioralMetrics,
    order: str,
    beta: Sequence[float],
    validation_question_ids: Sequence[str],
    selection_criterion: str | None = None,
    selection_split: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "split": "behavior_validation",
        "order_regime": order,
        "beta": list(beta),
        "neutral_accuracy": metrics.neutral_accuracy,
        "ftw": metrics.ftw,
        "cbr": metrics.cbr,
        "selectivity": metrics.selectivity,
        "pra_mean": metrics.pra_mean,
        "pra_all": metrics.pra_all,
        "n_q_plus": metrics.n_q_plus,
        "n_q_minus": metrics.n_q_minus,
        "validation_question_ids": list(validation_question_ids),
    }
    if selection_criterion is not None:
        payload["selection_criterion"] = selection_criterion
    if selection_split is not None:
        payload["selection_split"] = selection_split
    return payload


def discover_best_betas_by_criterion(opt_dir: Path | str) -> dict[str, tuple[float, ...]]:
    """Load opt-split best βs from best_checkpoint_by_*.json (DEC-100)."""
    directory = Path(opt_dir)
    found: dict[str, tuple[float, ...]] = {}
    for path in sorted(directory.glob("best_checkpoint_by_*.json")):
        match = _BEST_BY_RE.match(path.name)
        if match is None:
            continue
        metric = match.group(1)
        ckpt = load_checkpoint(json.loads(path.read_text(encoding="utf-8")))
        found[metric] = tuple(float(x) for x in ckpt["beta"])
    legacy = directory / "best_checkpoint.json"
    if "l_total" not in found and legacy.is_file():
        ckpt = load_checkpoint(json.loads(legacy.read_text(encoding="utf-8")))
        found["l_total"] = tuple(float(x) for x in ckpt["beta"])
    if not found:
        raise FileNotFoundError(
            f"missing best checkpoint at {legacy} "
            f"(and no best_checkpoint_by_*.json under {directory})"
        )
    return found


def _criterion_margins(
    eval_payload: Mapping[str, Any],
    metric: str,
    *,
    beta: Sequence[float],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    by_crit = eval_payload.get("margins_by_criterion")
    if isinstance(by_crit, Mapping) and metric in by_crit:
        entry = dict(by_crit[metric])
        try:
            return (
                dict(entry["neutral"]),
                dict(entry["ib"]),
                dict(entry["cb"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"margins_by_criterion[{metric!r}] missing {exc.args[0]} "
                "(DEC-100)"
            ) from exc
    if metric == "l_total":
        try:
            return (
                dict(eval_payload["current_neutral_margins"]),
                dict(eval_payload["current_ib_margins"]),
                dict(eval_payload["current_cb_margins"]),
            )
        except KeyError as exc:
            raise ValueError(
                "full_study requires current_* margins for l_total "
                "(or margins_by_criterion['l_total']; DEC-100)"
            ) from exc
    del beta
    raise ValueError(
        f"full_study requires margins_by_criterion[{metric!r}] "
        f"when best_checkpoint_by_{metric}.json exists (DEC-100)"
    )


def run_full_study_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    eval_payload: Mapping[str, Any],
    holdout_question_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate best β(s) and β=0 on behavior_validation; no holdout (DEC-098/100)."""
    if freeze_status != "sealed":
        raise HoldoutAccessError(
            "full_study requires freeze_status='sealed' "
            f"(got {freeze_status!r}; DEC-055 / DEC-069)"
        )
    if holdout_question_ids:
        # Ignore holdout IDs; never call the holdout loader from full_study.
        del holdout_question_ids

    opt_dir = Path(study.run.artifact_dir) / "optimize"
    best_betas = discover_best_betas_by_criterion(opt_dir)
    order = study_order_regime(study)
    baselines = dict(eval_payload["baseline_neutral_margins_by_order"])
    try:
        ni_neutral = dict(eval_payload["non_intervened_neutral_margins"])
        ni_ib = dict(eval_payload["non_intervened_ib_margins"])
        ni_cb = dict(eval_payload["non_intervened_cb_margins"])
    except KeyError as exc:
        raise ValueError(
            "full_study requires non_intervened_* margins in eval_payload "
            "(DEC-098 comparison log)"
        ) from exc
    epsilon = float(study.experiment.tie_band_epsilon)
    tie_policy = str(study.experiment.tie_policy)
    val_ids = list(eval_payload.get("validation_question_ids", ()))

    # Frozen partition from β=0 neutrals (shared across all criterion logs).
    partition_source = baselines.get(order)
    if partition_source is None:
        partition_source = ni_neutral
    partition = build_baseline_partition(
        order_regime=order,
        neutral_margins=partition_source,
        epsilon=epsilon,
        tie_policy=tie_policy,
    )
    artifact = freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash=study.stack.model.revision,
        prompt_template_hash="orch-full-study-prompt",
        order_manifest_hash=f"orch-full-study-order-{order}",
        dataset_manifest_hash="orch-full-study-dataset",
    )

    out_dir = Path(study.run.artifact_dir) / "full_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    metrics_out: dict[str, Any] = {
        "order_regime": order,
        "holdout_accessed": False,
    }

    # Criterion order: DEC-097 keys that are present, l_total last among equals
    # for stable writing; prefer declared tuple order.
    for metric in _LOSS_CRITERIA:
        if metric not in best_betas:
            continue
        beta = best_betas[metric]
        neutral, ib, cb = _criterion_margins(eval_payload, metric, beta=beta)
        metrics = compute_behavioral_metrics(
            frozen_partition=artifact,
            current_neutral_margins=neutral,
            current_ib_margins=ib,
            current_cb_margins=cb,
            epsilon=epsilon,
        )
        payload = _behavioral_payload(
            metrics=metrics,
            order=order,
            beta=beta,
            validation_question_ids=val_ids,
            selection_criterion=metric,
            selection_split="optimization",
        )
        by_path = out_dir / f"behavioral_best_by_{metric}.json"
        by_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts[f"behavioral_best_by_{metric}"] = str(by_path)
        metrics_out[f"{metric}_ftw"] = metrics.ftw
        metrics_out[f"{metric}_cbr"] = metrics.cbr
        metrics_out[f"{metric}_selectivity"] = metrics.selectivity
        log_progress(
            "full_study_behavioral_best_by",
            order_regime=order,
            selection_criterion=metric,
            ftw=metrics.ftw,
            cbr=metrics.cbr,
            selectivity=metrics.selectivity,
            path=str(by_path),
        )
        if metric == "l_total":
            # Legacy DEC-069/098 path name (same payload as best-by-l_total).
            behavioral_path = out_dir / "behavioral.json"
            behavioral_path.write_text(
                json.dumps(payload, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            artifacts["behavioral"] = str(behavioral_path)
            metrics_out["ftw"] = metrics.ftw
            metrics_out["cbr"] = metrics.cbr
            metrics_out["selectivity"] = metrics.selectivity
            log_progress(
                "full_study_behavioral",
                order_regime=order,
                ftw=metrics.ftw,
                cbr=metrics.cbr,
                selectivity=metrics.selectivity,
                path=str(behavioral_path),
            )

    ni_metrics = compute_behavioral_metrics(
        frozen_partition=artifact,
        current_neutral_margins=ni_neutral,
        current_ib_margins=ni_ib,
        current_cb_margins=ni_cb,
        epsilon=epsilon,
    )
    zero_beta = (
        tuple(0.0 for _ in best_betas["l_total"])
        if "l_total" in best_betas
        else (0.0,)
    )
    if "l_total" not in best_betas:
        # Any discovered beta length works for zero padding.
        any_beta = next(iter(best_betas.values()))
        zero_beta = tuple(0.0 for _ in any_beta) if any_beta else (0.0,)

    ni_path = out_dir / "behavioral_non_intervened.json"
    ni_payload = _behavioral_payload(
        metrics=ni_metrics,
        order=order,
        beta=zero_beta,
        validation_question_ids=val_ids,
    )
    ni_path.write_text(
        json.dumps(ni_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts["behavioral_non_intervened"] = str(ni_path)
    metrics_out["non_intervened_ftw"] = ni_metrics.ftw
    metrics_out["non_intervened_cbr"] = ni_metrics.cbr
    metrics_out["non_intervened_selectivity"] = ni_metrics.selectivity
    log_progress(
        "full_study_behavioral_non_intervened",
        order_regime=order,
        ftw=ni_metrics.ftw,
        cbr=ni_metrics.cbr,
        selectivity=ni_metrics.selectivity,
        path=str(ni_path),
    )
    return {
        "metrics": metrics_out,
        "artifacts": artifacts,
    }
