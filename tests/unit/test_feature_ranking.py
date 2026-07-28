"""Signed-Jacobian feature ranking (Phase F)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection import rank_suppression_candidates


@pytest.mark.unit
def test_feature_ranking__suppression_only__positive_jacobian_predicts_loss_reduction() -> (
    None
):
    """FEAT-008: rank descending signed J, never by magnitude.

    With dL ~ J_j * dbeta_j and dbeta_j < 0, a positive J predicts a lower
    loss. The strongly negative feature has the largest |J| and must rank
    last, not first.
    """
    scores = {
        (0, 7): 0.1,
        (0, 3): -5.0,
        (1, 2): 2.0,
        (1, 9): 0.0,
    }

    ranked = rank_suppression_candidates(signed_jacobians=scores)

    assert [(candidate.layer, candidate.feature_id) for candidate in ranked] == [
        (1, 2),
        (0, 7),
        (1, 9),
        (0, 3),
    ]

    delta_beta = -0.25
    for candidate in ranked:
        predicted_loss_change = candidate.signed_jacobian * delta_beta
        assert candidate.suppression_beneficial == (candidate.signed_jacobian > 0.0)
        if candidate.suppression_beneficial:
            assert predicted_loss_change < 0.0
        else:
            assert predicted_loss_change >= 0.0


@pytest.mark.unit
def test_feature_ranking__bidirectional__stores_preferred_coefficient_direction() -> (
    None
):
    """FEAT-009: the loss-decreasing direction is dbeta_j proportional to -J_j."""
    scores = {
        (0, 7): 0.1,
        (0, 3): -5.0,
        (1, 9): 0.0,
    }

    ranked = rank_suppression_candidates(signed_jacobians=scores)
    preferred = {
        (candidate.layer, candidate.feature_id): candidate.preferred_bidirectional_sign
        for candidate in ranked
    }

    assert preferred == {(0, 7): -1.0, (0, 3): 1.0, (1, 9): 0.0}

    # The bidirectional column is stored beside, not instead of, the
    # suppression-only order.
    assert [(candidate.layer, candidate.feature_id) for candidate in ranked] == [
        (0, 7),
        (1, 9),
        (0, 3),
    ]
