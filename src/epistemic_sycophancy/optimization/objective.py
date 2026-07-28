"""Deterministic optimizer-facing objective evaluation (OPT-001)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from epistemic_sycophancy.logging.trial_records import (
    ObjectiveComponents,
    build_objective_components,
)
from epistemic_sycophancy.objective.total import evaluate_objective


def evaluate_optimizer_objective(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    cb_margins_by_question: Mapping[object, Sequence[float]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    beta: Sequence[float],
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
) -> ObjectiveComponents:
    """Evaluate the Phase G objective and return DEC-026 logged components.

    Same β and regime must yield identical scalar and component values
    (OPT-001; mandatory for CMA-ES).
    """
    result = evaluate_objective(
        ib_margins_by_question=ib_margins_by_question,
        cb_margins_by_question=cb_margins_by_question,
        baseline_cb_margins=baseline_cb_margins,
        baseline_neutral_margins=baseline_neutral_margins,
        current_neutral_margins=current_neutral_margins,
        q_plus=q_plus,
        q_minus=q_minus,
        beta=beta,
        tau=tau,
        w_r=w_r,
        w_u=w_u,
        delta_n=delta_n,
        delta_c=delta_c,
        lambda_n=lambda_n,
        lambda_c=lambda_c,
        lambda_beta=lambda_beta,
    )
    return build_objective_components(
        result,
        lambda_n=lambda_n,
        lambda_c=lambda_c,
        lambda_beta=lambda_beta,
    )
