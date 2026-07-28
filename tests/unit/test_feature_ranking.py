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


@pytest.mark.unit
def test_feature_selection__fixed_artifacts__produce_stable_scores_and_tie_order() -> None:
    """FEAT-024: identical inputs → identical order; ties break ascending (layer, id)."""
    scores = {
        (1, 5): 1.0,
        (0, 9): 1.0,  # same J as (1,5); smaller layer wins the tie
        (0, 2): 1.0,  # same J; smaller feature_id within layer 0
        (2, 0): 0.5,
    }
    first = rank_suppression_candidates(signed_jacobians=scores)
    second = rank_suppression_candidates(signed_jacobians=scores)
    assert first == second
    assert [(c.layer, c.feature_id) for c in first] == [
        (0, 2),
        (0, 9),
        (1, 5),
        (2, 0),
    ]


@pytest.mark.unit
def test_feature_selection__answer_orders__produce_separate_jacobian_artifacts() -> None:
    """FEAT-031: CF, IF, and RO produce three distinct keyed artifacts.

    Scores must not overwrite one another and must not be silently averaged.
    """
    from epistemic_sycophancy.feature_selection import (
        FeatureSelectionArtifact,
        build_order_specific_artifacts,
    )

    row_template = {
        "layer": 0,
        "feature_id": 0,
        "absolute_sensitivity": 1.0,
        "raw_projection": 1.0,
        "mean_active_rate": 1.0,
        "feature_scale": 1.0,
        "suppression_beneficial": True,
        "preferred_bidirectional_sign": -1.0,
        "n_questions": 1,
        "n_prompts": 1,
    }
    # Distinct signed Jacobians per order; mean would be 0.
    by_order = {
        "CF": FeatureSelectionArtifact(
            rows=[{**row_template, "signed_jacobian": 3.0, "absolute_sensitivity": 3.0}],
            question_ids=frozenset({"q1"}),
            feature_selection_question_ids=frozenset({"q1"}),
        ),
        "IF": FeatureSelectionArtifact(
            rows=[{**row_template, "signed_jacobian": -3.0, "absolute_sensitivity": 3.0}],
            question_ids=frozenset({"q1"}),
            feature_selection_question_ids=frozenset({"q1"}),
        ),
        "RO": FeatureSelectionArtifact(
            rows=[{**row_template, "signed_jacobian": 1.5, "absolute_sensitivity": 1.5}],
            question_ids=frozenset({"q1"}),
            feature_selection_question_ids=frozenset({"q1"}),
        ),
    }
    artifacts = build_order_specific_artifacts(
        artifacts_by_order=by_order,
        component="resistance",
        model_revision_hash="model",
        sae_revision_hash="sae",
        scope="last_prompt_token",
        scale_source="decoder_norm",
        dataset_manifest_hash="dataset",
    )

    assert set(artifacts.keys()) == {"CF", "IF", "RO"}
    assert artifacts["CF"].rows[0].signed_jacobian == 3.0
    assert artifacts["IF"].rows[0].signed_jacobian == -3.0
    assert artifacts["RO"].rows[0].signed_jacobian == 1.5
    # No silent averaging: mean of CF and IF would be 0.
    assert artifacts["CF"].rows[0].signed_jacobian != artifacts["IF"].rows[0].signed_jacobian
    assert all(a.fingerprint for a in artifacts.values())
    assert len({a.fingerprint for a in artifacts.values()}) == 3
    assert artifacts["CF"].order_regime == "CF"
    assert artifacts["IF"].order_regime == "IF"
    assert artifacts["RO"].order_regime == "RO"
