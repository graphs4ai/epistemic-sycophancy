"""Structured prompt representation."""

from __future__ import annotations

import re
from dataclasses import dataclass

# MC option labels as emitted by the renderer (line-anchored "A. …" / "B. …").
_OPTION_LABEL_LINE = re.compile(r"(?m)^[ \t]*[AB]\.")

# Explicit letter-choice leakage ("the answer is A", "Pick B.", …).
_CHOICE_LETTER_LEAK = re.compile(
    r"(?i)\b(?:answer\s+is|pick|choose|select)\s+[AB]\b"
)


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

    Belief may state candidate *content*, but must not contain generated MC
    option labels (line-anchored ``A.`` / ``B.``), explicit choice-letter cues,
    or the literal answer suffix.

    Substring checks for ``\"A.\"`` / ``\"B.\"`` are forbidden: TruthfulQA
    beliefs include initials (``J. B. Rhine``) and words like ``DNA.`` (DEC-082).
    """
    if _OPTION_LABEL_LINE.search(belief_text) or _CHOICE_LETTER_LEAK.search(
        belief_text
    ):
        raise ValueError(
            "belief text must not contain answer-label artifacts 'A.' or 'B.'; "
            f"got {belief_text!r}"
        )
    if answer_suffix and answer_suffix in belief_text:
        raise ValueError(
            "belief text must not contain the literal answer suffix; "
            f"suffix={answer_suffix!r}, belief={belief_text!r}"
        )
