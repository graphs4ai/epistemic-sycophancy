"""Feature-selection loss components and their frozen subsets (Phase F)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from epistemic_sycophancy.metrics.baseline_partition import BaselinePartition

# Spec §11.2 / FEAT-010: condition and frozen question subset per component.
COMPONENT_CONDITION: dict[str, str] = {
    "resistance": "IB",
    "recovery": "CB",
    "neutral_surrogate": "N",
    "correct_surrogate": "CB",
}

COMPONENT_SUBSET: dict[str, str] = {
    "resistance": "q_plus",
    "recovery": "q_minus",
    "neutral_surrogate": "all_questions",
    "correct_surrogate": "q_plus",
}


class _PromptRow(Protocol):
    question_id: str
    condition: str


PromptRowT = TypeVar("PromptRowT", bound=_PromptRow)


def component_question_subset(
    *,
    component: str,
    partition: BaselinePartition,
    all_question_ids: Sequence[str],
) -> frozenset[str]:
    """Return the frozen question subset Q_u for a selection component.

    The subset is read from the frozen baseline partition and is never
    recomputed from current or intervened margins.
    """
    subset = COMPONENT_SUBSET[component]
    if subset == "q_plus":
        return partition.q_plus
    if subset == "q_minus":
        return partition.q_minus
    return frozenset(all_question_ids)


def selection_component_prompts(
    *,
    component: str,
    prompt_rows: Sequence[PromptRowT],
    partition: BaselinePartition,
) -> tuple[PromptRowT, ...]:
    """Return the prompts eligible for one feature-selection component."""
    condition = COMPONENT_CONDITION[component]
    questions = component_question_subset(
        component=component,
        partition=partition,
        all_question_ids=[row.question_id for row in prompt_rows],
    )
    return tuple(
        row
        for row in prompt_rows
        if row.condition == condition and row.question_id in questions
    )
