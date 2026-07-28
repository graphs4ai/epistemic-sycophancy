"""Full optimization objective assembly (Phase G)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
