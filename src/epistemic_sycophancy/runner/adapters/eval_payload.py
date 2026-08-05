"""Production eval_payload adapter for full_study (ORCH-025 / DEC-069 / DEC-100)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig, study_order_regime
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError

_LOSS_CRITERIA: tuple[str, ...] = (
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
)


def build_eval_payload(
    study: StudyConfig,
    stack: Any,
    *,
    best_beta: Sequence[float],
    validation_question_ids: Sequence[str],
    margin_scorer: Callable[..., Mapping[str, Any]] | None = None,
    holdout_question_ids: Sequence[str] = (),
    betas_by_criterion: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Score validation margins at best β(s) and β=0; never include holdout IDs.

    When ``betas_by_criterion`` is provided (DEC-100), score each distinct β once
    and populate ``margins_by_criterion``. Top-level ``current_*`` always mirrors
    the ``l_total`` (or ``best_beta``) intervened margins.
    """
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
        if not hasattr(stack, "score_belief_margins"):
            raise ValueError(
                "build_eval_payload requires margin_scorer or "
                "stack.score_belief_margins (DEC-076 live scoring)"
            )
        scorer = stack.score_belief_margins

    primary = tuple(float(b) for b in best_beta)
    criterion_betas: dict[str, tuple[float, ...]] = {}
    if betas_by_criterion:
        for key, beta in betas_by_criterion.items():
            metric = str(key)
            if metric not in _LOSS_CRITERIA:
                raise ValueError(
                    f"unsupported selection criterion {metric!r}; "
                    f"expected one of {_LOSS_CRITERIA}"
                )
            criterion_betas[metric] = tuple(float(x) for x in beta)
    if "l_total" not in criterion_betas:
        criterion_betas["l_total"] = primary
    elif criterion_betas["l_total"] != primary:
        # Prefer explicit criterion map; keep best_beta as documented primary.
        primary = criterion_betas["l_total"]

    zero = tuple(0.0 for _ in primary) if primary else (0.0,)

    def _score_scalar(
        belief: str, *, order: str, beta_vec: Sequence[float]
    ) -> dict[str, float]:
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

    def _score_seq(
        belief: str, *, order: str, beta_vec: Sequence[float]
    ) -> dict[str, tuple[float, ...]]:
        raw = dict(
            scorer(
                belief_condition=belief,
                question_ids=val_ids,
                beta=beta_vec,
                order_regime=order,
            )
        )
        out: dict[str, tuple[float, ...]] = {}
        for qid, value in raw.items():
            if qid in holdout:
                raise HoldoutAccessError(f"holdout id {qid!r} in eval margins")
            if isinstance(value, (list, tuple)):
                out[qid] = tuple(float(x) for x in value)
            else:
                out[qid] = (float(value),)
        return out

    order = study_order_regime(study)

    # Dedupe scoring by β tuple (DEC-100).
    unique_betas: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for beta_vec in (zero, *criterion_betas.values()):
        if beta_vec not in seen:
            seen.add(beta_vec)
            unique_betas.append(beta_vec)

    scored_n: dict[tuple[float, ...], dict[str, float]] = {}
    scored_ib: dict[tuple[float, ...], dict[str, tuple[float, ...]]] = {}
    scored_cb: dict[tuple[float, ...], dict[str, tuple[float, ...]]] = {}
    for beta_vec in unique_betas:
        scored_n[beta_vec] = _score_scalar("N", order=order, beta_vec=beta_vec)
        scored_ib[beta_vec] = _score_seq("IB", order=order, beta_vec=beta_vec)
        scored_cb[beta_vec] = _score_seq("CB", order=order, beta_vec=beta_vec)

    zero_n = scored_n[zero]
    zero_ib = scored_ib[zero]
    zero_cb = scored_cb[zero]
    current_n = scored_n[primary]
    current_ib = scored_ib[primary]
    current_cb = scored_cb[primary]
    baselines: dict[str, dict[str, float]] = {order: zero_n}

    margins_by_criterion: dict[str, dict[str, Any]] = {}
    for metric in _LOSS_CRITERIA:
        if metric not in criterion_betas:
            continue
        beta_vec = criterion_betas[metric]
        margins_by_criterion[metric] = {
            "beta": list(beta_vec),
            "neutral": scored_n[beta_vec],
            "ib": scored_ib[beta_vec],
            "cb": scored_cb[beta_vec],
        }

    return {
        "current_neutral_margins": current_n,
        "current_ib_margins": current_ib,
        "current_cb_margins": current_cb,
        "baseline_neutral_margins_by_order": baselines,
        "non_intervened_neutral_margins": zero_n,
        "non_intervened_ib_margins": zero_ib,
        "non_intervened_cb_margins": zero_cb,
        "margins_by_criterion": margins_by_criterion,
        "validation_question_ids": list(val_ids),
        "order_regime": order,
    }
