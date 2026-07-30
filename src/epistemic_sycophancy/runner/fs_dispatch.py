"""Feature-selection orchestration (ORCH-004 / DEC-060 / DEC-061 / DEC-085)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.feature_selection.components import COMPONENT_CONDITION
from epistemic_sycophancy.feature_selection.pool import build_common_feature_pool
from epistemic_sycophancy.runner.feature_selection import (
    run_feature_selection_stage_computed,
)

# Canonical §11.2 names (DEC-085); must match components.py.
_COMPONENTS = tuple(COMPONENT_CONDITION.keys())
_BEHAVIOR_COMPONENTS = ("resistance", "recovery")
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
    """Compute per-component order Jacobians, build pool, write artifact."""
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
        for component in _COMPONENTS:
            stage = run_feature_selection_stage_computed(
                order_regime=order,
                split_name="feature_selection",
                question_ids=qids,
                jacobian_fn=lambda *, order_regime, question_ids, _c=component: (
                    jacobian_fn(
                        order_regime=order_regime,
                        question_ids=question_ids,
                        component=_c,
                    )
                ),
                freeze_status=freeze_status,
                optimization_question_ids=optimization_question_ids,
                validation_question_ids=validation_question_ids,
                holdout_question_ids=holdout_question_ids,
            )
            lists[(order, component)] = dict(stage.signed_jacobians)
    # Fill missing order/component slots with empty maps so pool API is stable.
    for order in _ORDERS:
        for component in _COMPONENTS:
            lists.setdefault((order, component), {})

    # DEC-019: only resistance/recovery nominate into the common pool.
    behavior_lists = {
        key: scores
        for key, scores in lists.items()
        if key[1] in _BEHAVIOR_COMPONENTS
    }
    provisional_keys = sorted(
        {
            key
            for scores in behavior_lists.values()
            for key, value in scores.items()
            if float(value) > 0.0
        }
    )
    scales_map = dict(scale_fn(provisional_keys))
    pool = build_common_feature_pool(
        lists_by_order_and_component=behavior_lists,
        feature_scales=scales_map,
        pool_quota_per_list=int(study.experiment.pool_quota_per_list),
    )
    fingerprint = study_config_fingerprint(study)
    out_dir = Path(study.run.artifact_dir) / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "common_pool.json"
    provenance: dict[str, dict[str, object]] = {}
    for layer, fid in pool.feature_ids:
        key = f"{layer}:{fid}"
        nominators: list[dict[str, object]] = []
        for (order, component), scores in behavior_lists.items():
            if (layer, fid) in scores and float(scores[(layer, fid)]) > 0.0:
                nominators.append(
                    {
                        "order": order,
                        "component": component,
                        "signed_jacobian": float(scores[(layer, fid)]),
                    }
                )
        surrogates: dict[str, float] = {}
        for surr in ("neutral_surrogate", "correct_surrogate"):
            # Prefer CF annotation when present; else first nonempty order.
            value = None
            for order in regimes:
                scores = lists.get((order, surr), {})
                if (layer, fid) in scores:
                    value = float(scores[(layer, fid)])
                    break
            if value is not None:
                surrogates[surr] = value
        provenance[key] = {"nominators": nominators, "surrogates": surrogates}
    payload = {
        "schema_version": 2,
        "feature_ids": [[layer, fid] for layer, fid in pool.feature_ids],
        "feature_scales": list(pool.scales),
        "pool_size": len(pool.feature_ids),
        "scale_source": "decoder_norm",
        "study_yaml_fingerprint": fingerprint,
        "question_ids": list(qids),
        "split_name": "feature_selection",
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "pool": pool,
        "component_jacobians": lists,
        "metrics": {
            "pool_size": len(pool.feature_ids),
            "scale_source": "decoder_norm",
            "n_questions": len(qids),
        },
        "artifacts": {"pool": str(path)},
    }
