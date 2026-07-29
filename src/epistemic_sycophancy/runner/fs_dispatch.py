"""Feature-selection orchestration (ORCH-004 / DEC-060 / DEC-061)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.feature_selection.pool import build_common_feature_pool
from epistemic_sycophancy.runner.feature_selection import (
    run_feature_selection_stage_computed,
)

_COMPONENTS = ("resistance", "recovery", "neutral_preservation", "correct_belief")
_ORDERS = ("CF", "IF", "RO")


def run_feature_selection_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    jacobian_fn: Callable[..., Mapping[tuple[int, int], float]],
    scale_fn: Callable[[Sequence[tuple[int, int]]], Mapping[tuple[int, int], float]],
    question_ids: Sequence[str] | None = None,
    optimization_question_ids: Sequence[str] = (),
    validation_question_ids: Sequence[str] = (),
    holdout_question_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Compute order Jacobians, build pool with decoder_norm scales, write artifact."""
    smoke = study.run.smoke
    if question_ids is not None:
        qids = tuple(str(q) for q in question_ids)
    elif smoke.question_ids is not None:
        qids = tuple(smoke.question_ids)
    else:
        raise ValueError(
            "feature_selection dispatch requires question_ids when smoke uses "
            "n_questions without corpus injection"
        )

    regimes = tuple(study.run.order_regimes) or ("CF",)
    lists: dict[tuple[str, str], dict[tuple[int, int], float]] = {}
    for order in regimes:
        stage = run_feature_selection_stage_computed(
            order_regime=order,
            split_name="feature_selection",
            question_ids=qids,
            jacobian_fn=jacobian_fn,
            freeze_status=freeze_status,
            optimization_question_ids=optimization_question_ids,
            validation_question_ids=validation_question_ids,
            holdout_question_ids=holdout_question_ids,
        )
        # Replicate signed map across components for pool union (DEC-019 lists).
        for component in _COMPONENTS:
            lists[(order, component)] = dict(stage.signed_jacobians)
    # Fill missing order/component slots with empty maps so pool API is stable.
    for order in _ORDERS:
        for component in _COMPONENTS:
            lists.setdefault((order, component), {})

    # Provisional keys for scales: union of positive Jacobians seen.
    provisional_keys = sorted(
        {
            key
            for scores in lists.values()
            for key, value in scores.items()
            if float(value) > 0.0
        }
    )
    scales_map = dict(scale_fn(provisional_keys))
    pool = build_common_feature_pool(
        lists_by_order_and_component=lists,
        feature_scales=scales_map,
        pool_quota_per_list=int(study.experiment.pool_quota_per_list),
    )
    fingerprint = study_config_fingerprint(study)
    out_dir = Path(study.run.artifact_dir) / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "common_pool.json"
    payload = {
        "feature_ids": [[layer, fid] for layer, fid in pool.feature_ids],
            "feature_scales": list(pool.scales),
        "pool_size": len(pool.feature_ids),
        "scale_source": "decoder_norm",
        "study_yaml_fingerprint": fingerprint,
        "question_ids": list(qids),
        "split_name": "feature_selection",
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "pool": pool,
        "metrics": {
            "pool_size": len(pool.feature_ids),
            "scale_source": "decoder_norm",
            "n_questions": len(qids),
        },
        "artifacts": {"pool": str(path)},
    }
