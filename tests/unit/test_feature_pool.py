"""Candidate-pool eligibility for suppression (Phase F FEAT-025+)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection import (
    annotate_preservation_jacobians,
    eligible_suppression_candidates,
    rank_suppression_candidates,
)


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


@pytest.mark.unit
def test_feature_pool__selected_behavior_features__retain_signed_preservation_jacobians() -> (
    None
):
    """FEAT-026: preservation Jacobians annotate selected features; never rank them.

    Behavior ranking uses resistance/recovery signed Jacobians only. Neutral and
    correct-surrogate sensitivities are retained as annotations for later
    filtering/interpretation (DEC-019).
    """
    behavior_scores = {
        (0, 1): 2.0,
        (0, 2): -3.0,  # large |J| but nonpositive → excluded by default
        (1, 4): 0.5,
    }
    neutral_jacobians = {
        (0, 1): -1.5,
        (0, 2): 9.0,  # would dominate if mixed into behavior rank
        (1, 4): 0.25,
    }
    correct_surrogate_jacobians = {
        (0, 1): 0.8,
        (0, 2): -4.0,
        (1, 4): -0.1,
    }

    eligible = eligible_suppression_candidates(
        signed_jacobians=behavior_scores,
        pool_eligibility_override=False,
    )
    annotated = annotate_preservation_jacobians(
        candidates=eligible.candidates,
        neutral_jacobians=neutral_jacobians,
        correct_surrogate_jacobians=correct_surrogate_jacobians,
    )

    assert [(c.layer, c.feature_id) for c in annotated] == [(0, 1), (1, 4)]
    by_key = {(c.layer, c.feature_id): c for c in annotated}
    assert by_key[(0, 1)].neutral_jacobian == -1.5
    assert by_key[(0, 1)].correct_surrogate_jacobian == 0.8
    assert by_key[(1, 4)].neutral_jacobian == 0.25
    assert by_key[(1, 4)].correct_surrogate_jacobian == -0.1

    # Behavior rank ignores preservation scores: mixing them into ranking must
    # not change the behavior order, and (0,2) stays out despite huge neutral J.
    behavior_rank = rank_suppression_candidates(signed_jacobians=behavior_scores)
    assert [(c.layer, c.feature_id) for c in behavior_rank] == [
        (0, 1),
        (1, 4),
        (0, 2),
    ]
    mixed_would_invert = rank_suppression_candidates(
        signed_jacobians={
            key: behavior_scores[key] + neutral_jacobians[key]
            for key in behavior_scores
        }
    )
    assert [(c.layer, c.feature_id) for c in mixed_would_invert] != [
        (c.layer, c.feature_id) for c in behavior_rank
    ]
    assert (0, 2) not in {(c.layer, c.feature_id) for c in annotated}
