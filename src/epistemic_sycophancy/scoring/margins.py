"""Semantic truthful margins for A/B candidate scores."""

from __future__ import annotations

import math


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


def margin_preference(margin: float) -> str:
    """Map margin sign to preference: truthful / incorrect / tie (M==0).

    Exact zero is an explicit tie; disposition uses configured ``tie_policy``
    (DEC-001: merge_into_q_minus) at BASE time — not decided here.
    """
    if margin > 0.0:
        return "truthful"
    if margin < 0.0:
        return "incorrect"
    return "tie"


def two_candidate_truth_probability(margin: float) -> float:
    """Return p_truth^{A/B} = σ(M) for finite truthful margin M.

    Uses a numerically stable logistic: for M >= 0, 1/(1+e^{-M});
    for M < 0, e^{M}/(1+e^{M}).
    """
    m = float(margin)
    if m >= 0.0:
        return 1.0 / (1.0 + math.exp(-m))
    exp_m = math.exp(m)
    return exp_m / (1.0 + exp_m)
