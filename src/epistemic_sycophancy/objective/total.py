"""Full optimization objective assembly (Phase G)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from epistemic_sycophancy.objective.aggregation import question_macro_mean
from epistemic_sycophancy.objective.losses import logistic_margin_loss


def resistance_prompt_losses(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> dict[object, list[float]]:
    """Return φ(M) for each IB prompt of each question in Q+.

    Only q∈Q+ and IB variants are eligible (OBJ-001).
    """
    q_plus_set = frozenset(q_plus)
    return {
        question_id: [
            logistic_margin_loss(float(margin), tau=tau) for margin in margins
        ]
        for question_id, margins in ib_margins_by_question.items()
        if question_id in q_plus_set
    }


def resistance_loss(
    *,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> float:
    """Question-macro resistance: mean within question, then across Q+."""
    return question_macro_mean(
        resistance_prompt_losses(
            ib_margins_by_question=ib_margins_by_question,
            q_plus=q_plus,
            tau=tau,
        )
    )


def recovery_prompt_losses(
    *,
    cb_margins_by_question: Mapping[object, Sequence[float]],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> dict[object, list[float]]:
    """Return φ(M) for each CB prompt of each question in Q-."""
    q_minus_set = frozenset(q_minus)
    return {
        question_id: [
            logistic_margin_loss(float(margin), tau=tau) for margin in margins
        ]
        for question_id, margins in cb_margins_by_question.items()
        if question_id in q_minus_set
    }


def recovery_loss(
    *,
    cb_margins_by_question: Mapping[object, Sequence[float]],
    q_minus: frozenset[object] | set[object] | Sequence[object],
    tau: float,
) -> float:
    """Question-macro recovery: mean within question, then across Q-."""
    return question_macro_mean(
        recovery_prompt_losses(
            cb_margins_by_question=cb_margins_by_question,
            q_minus=q_minus,
            tau=tau,
        )
    )


def behavioral_loss(
    *,
    l_resist: float,
    l_recover: float,
    w_r: float,
    w_u: float,
) -> float:
    """L_behavior = w_R L_resist + w_U L_recover (explicit weights, not subset sizes)."""
    return float(w_r) * float(l_resist) + float(w_u) * float(l_recover)
