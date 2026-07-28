"""Full optimization objective assembly (Phase G)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from epistemic_sycophancy.objective.aggregation import question_macro_mean
from epistemic_sycophancy.objective.losses import (
    baseline_relative_hinge,
    logistic_margin_loss,
)


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


def neutral_question_penalties(
    *,
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    delta_n: float,
) -> dict[object, float]:
    """Per-question neutral hinge d_q,N = [M0 - M(β) - δ_N]_+."""
    return {
        question_id: float(
            baseline_relative_hinge(
                baseline_margin=float(baseline_neutral_margins[question_id]),
                current_margin=float(current_neutral_margins[question_id]),
                delta=delta_n,
            )
        )
        for question_id in baseline_neutral_margins
    }


def neutral_preservation_loss(
    *,
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    delta_n: float,
) -> float:
    """L_neutral = (1/|Q|) Σ_q d_q,N over the full optimization question set."""
    penalties = neutral_question_penalties(
        baseline_neutral_margins=baseline_neutral_margins,
        current_neutral_margins=current_neutral_margins,
        delta_n=delta_n,
    )
    if not penalties:
        raise ValueError("neutral_preservation_loss requires at least one question")
    return sum(penalties.values()) / float(len(penalties))


def correct_belief_question_penalties(
    *,
    baseline_cb_margins: Mapping[object, Sequence[float]],
    current_cb_margins: Mapping[object, Sequence[float]],
    q_plus: frozenset[object] | set[object] | Sequence[object],
    delta_c: float,
) -> dict[object, list[float]]:
    """Per-variant CB hinges for each q∈Q+: [M0 - M(β) - δ_C]_+."""
    q_plus_set = frozenset(q_plus)
    result: dict[object, list[float]] = {}
    for question_id in q_plus_set:
        baselines = baseline_cb_margins[question_id]
        currents = current_cb_margins[question_id]
        if len(baselines) != len(currents):
            raise ValueError(
                f"baseline/current CB margin length mismatch for {question_id!r}"
            )
        result[question_id] = [
            float(
                baseline_relative_hinge(
                    baseline_margin=float(b0),
                    current_margin=float(m),
                    delta=delta_c,
                )
            )
            for b0, m in zip(baselines, currents, strict=True)
        ]
    return result
