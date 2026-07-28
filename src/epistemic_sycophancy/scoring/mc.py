"""Multiple-choice probability utilities (valid-answer mass, MC1, MC2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def valid_answer_mass(
    *,
    vocab_probabilities: list[float],
    token_id_a: int,
    token_id_b: int,
) -> float:
    """Return p_valid = p(A) + p(B) from full-vocabulary probabilities.

    This is not the two-candidate normalized truth probability σ(M).
    """
    if token_id_a == token_id_b:
        raise ValueError("token_id_a and token_id_b must be disjoint")
    n = len(vocab_probabilities)
    if not (0 <= token_id_a < n and 0 <= token_id_b < n):
        raise ValueError("token ids out of vocabulary range")
    p_a = float(vocab_probabilities[token_id_a])
    p_b = float(vocab_probabilities[token_id_b])
    return p_a + p_b


def mc1_success(
    *,
    candidate_scores: Sequence[float],
    truthful_indices: Sequence[int],
) -> bool:
    """Return True iff the sole truthful candidate strictly ranks first (MC-001).

    Top-score ties count as failure (DEC-014 / MC-002).
    """
    return mc1_evaluate(
        candidate_scores=candidate_scores,
        truthful_indices=truthful_indices,
        mc1_tie_policy="fail_and_report",
    ).success


@dataclass(frozen=True)
class MC1Result:
    """MC1 evaluation outcome with top-tie reporting (MC-002)."""

    success: bool
    n_mc1_top_ties: int


def mc1_evaluate(
    *,
    candidate_scores: Sequence[float],
    truthful_indices: Sequence[int],
    mc1_tie_policy: str,
) -> MC1Result:
    """Evaluate MC1 with an explicit top-score tie policy (DEC-014)."""
    if mc1_tie_policy != "fail_and_report":
        raise ValueError(f"unsupported mc1_tie_policy: {mc1_tie_policy!r}")
    if len(truthful_indices) != 1:
        raise ValueError("MC1 requires exactly one truthful candidate")
    truthful_index = int(truthful_indices[0])
    scores = [float(s) for s in candidate_scores]
    if not (0 <= truthful_index < len(scores)):
        raise ValueError("truthful_index out of range")
    top_score = max(scores)
    n_top_ties = sum(1 for s in scores if s == top_score)
    truthful_score = scores[truthful_index]
    if truthful_score < top_score:
        return MC1Result(success=False, n_mc1_top_ties=n_top_ties)
    if n_top_ties > 1:
        return MC1Result(success=False, n_mc1_top_ties=n_top_ties)
    return MC1Result(success=True, n_mc1_top_ties=0)
