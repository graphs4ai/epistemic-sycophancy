"""Question-macro aggregation utilities (Phase D)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def question_macro_mean(
    values_by_question: Mapping[object, Sequence[float]],
) -> float:
    """Mean within each question, then mean across questions.

    Each question receives equal weight regardless of how many values it has.

    Args:
        values_by_question: Mapping from question_id to per-prompt scalar values.

    Returns:
        The question-macro mean as a float.
    """
    if not values_by_question:
        raise ValueError("values_by_question must be non-empty")
    question_means: list[float] = []
    for values in values_by_question.values():
        if not values:
            raise ValueError("each question must have at least one value")
        question_means.append(sum(float(v) for v in values) / len(values))
    return sum(question_means) / len(question_means)
