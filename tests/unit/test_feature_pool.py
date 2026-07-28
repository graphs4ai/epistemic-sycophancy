"""Candidate-pool eligibility for suppression (Phase F FEAT-025)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection import eligible_suppression_candidates


@pytest.mark.unit
def test_feature_pool__suppression_only__excludes_nonpositive_behavior_jacobians_by_default() -> (
    None
):
    """FEAT-025: default eligibility is signed_jacobian > 0 (DEC-019)."""
    scores = {
        (0, 1): 2.0,
        (0, 2): 0.0,
        (1, 3): -1.0,
        (1, 4): 0.5,
    }
    eligible = eligible_suppression_candidates(
        signed_jacobians=scores,
        pool_eligibility_override=False,
    )
    assert eligible.pool_eligibility_override is False
    assert [(c.layer, c.feature_id) for c in eligible.candidates] == [(0, 1), (1, 4)]
    assert all(c.signed_jacobian > 0.0 for c in eligible.candidates)

    overridden = eligible_suppression_candidates(
        signed_jacobians=scores,
        pool_eligibility_override=True,
    )
    assert overridden.pool_eligibility_override is True
    assert len(overridden.candidates) == 4
    assert [(c.layer, c.feature_id) for c in overridden.candidates] == [
        (0, 1),
        (1, 4),
        (0, 2),
        (1, 3),
    ]
