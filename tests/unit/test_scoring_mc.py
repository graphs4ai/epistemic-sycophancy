"""Valid-answer mass tests (Phase C SCORE-010)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.scoring.margins import two_candidate_truth_probability
from epistemic_sycophancy.scoring.mc import valid_answer_mass


@pytest.mark.unit
def test_scoring__valid_answer_mass__is_computed_from_full_model_probabilities() -> None:
    """SCORE-010: p_valid = p(A)+p(B) from full vocab; distinct from σ(M)."""
    # Full-vocab toy distribution: mass outside {A,B}.
    # Indices 65=A, 66=B (DEC-010 ascii_letter).
    vocab_probs = [0.0] * 80
    vocab_probs[65] = 0.20  # A
    vocab_probs[66] = 0.15  # B
    vocab_probs[10] = 0.65  # other mass
    assert abs(sum(vocab_probs) - 1.0) < 1e-12

    p_valid = valid_answer_mass(
        vocab_probabilities=vocab_probs,
        token_id_a=65,
        token_id_b=66,
    )
    assert p_valid == pytest.approx(0.35, abs=1e-12, rel=1e-12)
    assert 0.0 <= p_valid <= 1.0 + 1e-12

    # Must not equal two-candidate normalized truth probability.
    # For logits/scores where σ(M) uses score difference, use semantic scores:
    # If we wrongly used σ on log-odds of A vs B alone we'd get 0.20/(0.20+0.15).
    two_cand_normalized = 0.20 / (0.20 + 0.15)
    assert p_valid != pytest.approx(two_cand_normalized, abs=1e-12, rel=1e-12)
    # And σ(M) for M = log(0.20)-log(0.15) equals that normalized ratio — still ≠ p_valid.
    import math

    margin = math.log(0.20) - math.log(0.15)
    assert two_candidate_truth_probability(margin) == pytest.approx(
        two_cand_normalized, abs=1e-12, rel=1e-12
    )
    assert p_valid != pytest.approx(
        two_candidate_truth_probability(margin), abs=1e-12, rel=1e-12
    )
