"""Feature-selection score artifacts and leakage gates (Phase F)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection import (
    FeatureSelectionArtifact,
    HoldoutAccessError,
)


@pytest.mark.unit
def test_feature_artifact__stores_raw_projection_active_rate_scale_and_normalized_jacobian() -> (
    None
):
    """FEAT-027: separate decoder alignment, prevalence, scale, and signed J.

    Hand values: h=2.0, mean_active_rate=0.5, s=4.0 → signed_jacobian = s * rate * h
    is not required here; the artifact stores each diagnostic column independently
    so a high rank can be attributed to alignment vs prevalence vs scale.
    """
    rows = (
        {
            "layer": 0,
            "feature_id": 1,
            "signed_jacobian": 4.0,
            "absolute_sensitivity": 4.0,
            "raw_projection": 2.0,
            "mean_active_rate": 0.5,
            "feature_scale": 4.0,
            "suppression_beneficial": True,
            "preferred_bidirectional_sign": -1.0,
            "n_questions": 3,
            "n_prompts": 7,
        },
        {
            "layer": 1,
            "feature_id": 0,
            "signed_jacobian": -1.5,
            "absolute_sensitivity": 1.5,
            "raw_projection": -3.0,
            "mean_active_rate": 1.0,
            "feature_scale": 0.5,
            "suppression_beneficial": False,
            "preferred_bidirectional_sign": 1.0,
            "n_questions": 3,
            "n_prompts": 7,
        },
    )
    artifact = FeatureSelectionArtifact(rows=rows)

    assert len(artifact.rows) == 2
    first = artifact.rows[0]
    assert first.signed_jacobian == 4.0
    assert first.absolute_sensitivity == 4.0
    assert first.raw_projection == 2.0
    assert first.mean_active_rate == 0.5
    assert first.feature_scale == 4.0
    assert first.suppression_beneficial is True
    assert first.preferred_bidirectional_sign == -1.0
    assert first.n_questions == 3
    assert first.n_prompts == 7

    second = artifact.rows[1]
    assert second.signed_jacobian == -1.5
    assert second.raw_projection == -3.0
    assert second.mean_active_rate == 1.0
    assert second.feature_scale == 0.5
    assert second.suppression_beneficial is False
    assert second.preferred_bidirectional_sign == 1.0


@pytest.mark.unit
def test_feature_selection__data_access__is_limited_to_feature_selection_split() -> None:
    """FEAT-028: question IDs outside feature_selection raise HoldoutAccessError."""
    feature_selection_ids = frozenset({"q1", "q2", "q3"})
    rows = (
        {
            "layer": 0,
            "feature_id": 0,
            "signed_jacobian": 1.0,
            "absolute_sensitivity": 1.0,
            "raw_projection": 1.0,
            "mean_active_rate": 1.0,
            "feature_scale": 1.0,
            "suppression_beneficial": True,
            "preferred_bidirectional_sign": -1.0,
            "n_questions": 2,
            "n_prompts": 2,
        },
    )

    ok = FeatureSelectionArtifact(
        rows=rows,
        question_ids=frozenset({"q1", "q2"}),
        feature_selection_question_ids=feature_selection_ids,
    )
    assert ok.question_ids <= feature_selection_ids

    with pytest.raises(HoldoutAccessError):
        FeatureSelectionArtifact(
            rows=rows,
            question_ids=frozenset({"q1", "holdout_q99"}),
            feature_selection_question_ids=feature_selection_ids,
        )

    with pytest.raises(HoldoutAccessError):
        FeatureSelectionArtifact(
            rows=rows,
            question_ids=frozenset({"optimization_q1"}),
            feature_selection_question_ids=feature_selection_ids,
        )
