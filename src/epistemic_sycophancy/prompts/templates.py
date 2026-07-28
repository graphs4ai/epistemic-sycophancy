"""Structured prompt representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StructuredPrompt:
    """Structured prompt fields before string rendering.

    Conditions for the same question/order/format must share every field
    except ``belief_context`` (and the ``belief_condition`` label).
    """

    question_text: str
    candidate_a: str
    candidate_b: str
    instruction: str
    suffix: str
    prompt_template_version: str
    format: str
    order_regime: str
    belief_condition: str
    belief_context: str | None


def assert_belief_text_has_no_label_artifacts(
    belief_text: str,
    *,
    answer_suffix: str | None = None,
) -> None:
    """Reject belief text that leaks answer labels or the template answer suffix.

    Belief may state candidate *content*, but must not contain generated labels
    such as ``\"A.\"`` / ``\"B.\"`` or the literal answer suffix.
    """
    if "A." in belief_text or "B." in belief_text:
        raise ValueError(
            "belief text must not contain answer-label artifacts 'A.' or 'B.'; "
            f"got {belief_text!r}"
        )
    if answer_suffix and answer_suffix in belief_text:
        raise ValueError(
            "belief text must not contain the literal answer suffix; "
            f"suffix={answer_suffix!r}, belief={belief_text!r}"
        )
