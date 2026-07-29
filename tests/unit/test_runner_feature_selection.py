"""RUN-010: feature-selection stage emits (layer, feature_id) keys for one order."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.runner.feature_selection import run_feature_selection_stage


@pytest.mark.unit
def test_runner__feature_selection_stage__emits_per_layer_keys_for_one_order() -> None:
    """RUN-010: CF-only FS stage returns Jacobians keyed by (layer, feature_id)."""
    fake_component_maps = {
        ("CF", "resistance"): {(9, 1): 0.5, (17, 3): 1.2, (9, 2): -0.1},
        ("CF", "recovery"): {(9, 1): 0.2, (22, 0): 0.8},
    }
    result = run_feature_selection_stage(
        order_regime="CF",
        split_name="feature_selection",
        component_jacobians=fake_component_maps,
        freeze_status="unsealed",
    )
    assert result.order_regime == "CF"
    assert result.split_name == "feature_selection"
    for key in result.signed_jacobians:
        assert isinstance(key, tuple) and len(key) == 2
        layer, feature_id = key
        assert isinstance(layer, int) and isinstance(feature_id, int)
    # Flat ints must not appear as keys.
    assert all(not isinstance(k, int) for k in result.signed_jacobians)

    with pytest.raises(Exception):
        run_feature_selection_stage(
            order_regime="CF",
            split_name="holdout_test_behavior",
            component_jacobians=fake_component_maps,
            freeze_status="unsealed",
        )


@pytest.mark.unit
def test_runner__feature_selection_stage__computes_jacobian_keys_on_tiny_fs_subset() -> None:
    """WIRE-007: jacobian_fn computes (layer, feature_id) on FS only (DEC-060)."""
    from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
    from epistemic_sycophancy.runner.feature_selection import (
        run_feature_selection_stage_computed,
    )

    seen: list[tuple[str, tuple[str, ...]]] = []

    def jacobian_fn(*, order_regime: str, question_ids: tuple[str, ...]):
        seen.append((order_regime, question_ids))
        return {(17, 3): 1.5, (17, 1): 0.25}

    result = run_feature_selection_stage_computed(
        order_regime="CF",
        split_name="feature_selection",
        question_ids=("q1", "q2"),
        jacobian_fn=jacobian_fn,
        freeze_status="unsealed",
        optimization_question_ids=("q10",),
        validation_question_ids=("q20",),
        holdout_question_ids=("q99",),
    )
    assert seen == [("CF", ("q1", "q2"))]
    assert set(result.signed_jacobians) == {(17, 3), (17, 1)}
    assert all(isinstance(k, tuple) and len(k) == 2 for k in result.signed_jacobians)

    with pytest.raises(HoldoutAccessError):
        run_feature_selection_stage_computed(
            order_regime="CF",
            split_name="optimization",
            question_ids=("q10",),
            jacobian_fn=jacobian_fn,
            freeze_status="unsealed",
            optimization_question_ids=("q10",),
            validation_question_ids=("q20",),
            holdout_question_ids=("q99",),
        )
