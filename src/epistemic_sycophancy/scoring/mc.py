"""Multiple-choice probability utilities (valid-answer mass)."""

from __future__ import annotations


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
