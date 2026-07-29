"""Production eval_payload adapter for full_study (ORCH-025 / DEC-069)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError


def build_eval_payload(
    study: StudyConfig,
    stack: Any,
    *,
    best_beta: Sequence[float],
    validation_question_ids: Sequence[str],
    margin_scorer: Callable[..., Mapping[str, Any]] | None = None,
    holdout_question_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Score behavior_validation margins at best β; never include holdout IDs."""
    del stack
    val_ids = tuple(str(q) for q in validation_question_ids)
    holdout = {str(q) for q in holdout_question_ids}
    if holdout and set(val_ids) & holdout:
        raise HoldoutAccessError(
            "build_eval_payload must not use holdout question IDs "
            f"(overlap={sorted(set(val_ids) & holdout)})"
        )
    if any(qid.startswith("holdout") for qid in val_ids):
        raise HoldoutAccessError("validation_question_ids look like holdout IDs")

    scorer = margin_scorer
    if scorer is None:
        raise ValueError(
            "build_eval_payload requires margin_scorer or stack.score_belief_margins"
        )

    beta = tuple(float(b) for b in best_beta)
    zero = tuple(0.0 for _ in beta) if beta else (0.0,)

    def _score(belief: str, *, order: str, beta_vec: Sequence[float]) -> dict[str, float]:
        raw = dict(
            scorer(
                belief_condition=belief,
                question_ids=val_ids,
                beta=beta_vec,
                order_regime=order,
            )
        )
        out: dict[str, float] = {}
        for qid, value in raw.items():
            if qid in holdout:
                raise HoldoutAccessError(f"holdout id {qid!r} in eval margins")
            if isinstance(value, (list, tuple)):
                out[qid] = float(value[0]) if value else 0.0
            else:
                out[qid] = float(value)
        return out

    current_n = _score("N", order="CF", beta_vec=beta)
    current_ib = _score("IB", order="CF", beta_vec=beta)
    current_cb = _score("CB", order="CF", beta_vec=beta)
    baselines: dict[str, dict[str, float]] = {}
    for order in study.run.order_regimes:
        baselines[str(order)] = _score("N", order=str(order), beta_vec=zero)

    return {
        "current_neutral_margins": current_n,
        "current_ib_margins": current_ib,
        "current_cb_margins": current_cb,
        "baseline_neutral_margins_by_order": baselines,
        "validation_question_ids": list(val_ids),
    }
