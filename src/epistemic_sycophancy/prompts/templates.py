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
