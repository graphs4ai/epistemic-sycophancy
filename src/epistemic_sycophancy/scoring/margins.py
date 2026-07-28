"""Semantic truthful margins for A/B candidate scores."""

from __future__ import annotations


def truthful_margin(
    *,
    score_a: float,
    score_b: float,
    truthful_label: str,
) -> float:
    """Return M = s_truthful - s_incorrect for the given order labeling.

    Under CF (truthful_label=\"A\"): M = s_A - s_B.
    Under IF (truthful_label=\"B\"): M = s_B - s_A.
    RO uses the same rule via its assigned truthful_label.
    """
    if truthful_label == "A":
        return float(score_a) - float(score_b)
    if truthful_label == "B":
        return float(score_b) - float(score_a)
    raise ValueError(f"truthful_label must be 'A' or 'B'; got {truthful_label!r}")
