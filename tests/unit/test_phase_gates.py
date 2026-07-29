"""Phase-gate / reproducibility tests (Phase I REPRO)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows


@pytest.mark.unit
def test_phase_gate__before_freeze__holdout_loader_raises_access_error() -> None:
    """REPRO-002: before frozen config, holdout loader raises HoldoutAccessError."""
    with pytest.raises(HoldoutAccessError):
        load_holdout_rows(freeze_status="unfrozen")


@pytest.mark.unit
def test_phase_gate__failed_identity_test__blocks_optimization() -> None:
    """REPRO-004: failed SAE identity suite blocks optimization entry."""
    from epistemic_sycophancy.reproducibility.phase_gates import (
        OptimizationBlockedError,
        require_identity_gate,
    )

    with pytest.raises(OptimizationBlockedError):
        require_identity_gate(identity_passed=False)


@pytest.mark.unit
def test_phase_gate__missing_or_mismatched_baseline_partition__blocks_optimization() -> None:
    """REPRO-005: missing or mismatched baseline partition blocks optimization."""
    from epistemic_sycophancy.reproducibility.phase_gates import (
        OptimizationBlockedError,
        require_baseline_partition_gate,
    )

    with pytest.raises(OptimizationBlockedError):
        require_baseline_partition_gate(
            expected_fingerprint="abc",
            actual_fingerprint=None,
        )
    with pytest.raises(OptimizationBlockedError):
        require_baseline_partition_gate(
            expected_fingerprint="abc",
            actual_fingerprint="xyz",
        )


@pytest.mark.unit
def test_phase_gate__feature_selection_artifact__cannot_reference_optimization_validation_or_holdout_rows() -> None:
    """REPRO-006: feature artifact question IDs must be feature_selection-only."""
    from epistemic_sycophancy.reproducibility.phase_gates import (
        require_feature_selection_split_gate,
    )

    with pytest.raises(HoldoutAccessError):
        require_feature_selection_split_gate(
            artifact_question_ids={"q_fs", "q_opt"},
            feature_selection_question_ids={"q_fs"},
            optimization_question_ids={"q_opt"},
            validation_question_ids={"q_val"},
            holdout_question_ids={"q_hold"},
        )


@pytest.mark.unit
def test_phase_gate__validation_selection__cannot_reference_holdout_rows() -> None:
    """REPRO-007: checkpoint/behavior selection cannot see holdout metrics."""
    from epistemic_sycophancy.optimization.selection import select_best_checkpoint

    with pytest.raises(HoldoutAccessError):
        select_best_checkpoint(
            [
                {
                    "checkpoint_id": "c1",
                    "trial_index": 0,
                    "l_total": 1.0,
                    "holdout_l_total": 0.1,
                }
            ]
        )


@pytest.mark.unit
def test_phase_gate__result_artifact__includes_required_hashes() -> None:
    """REPRO-001: result artifacts include the required hash/ID completeness set."""
    from epistemic_sycophancy.reproducibility.artifacts import (
        REQUIRED_RESULT_HASH_FIELDS,
        build_result_artifact_hashes,
    )

    hashes = build_result_artifact_hashes(
        dataset_manifest_hash="d1",
        prompt_template_hash="p1",
        order_manifest_hash="o1",
        model_revision="m1",
        tokenizer_revision="t1",
        sae_revision="s1",
        hook_configuration_hash="h1",
        selected_features_hash="f1",
        feature_scales_hash="fs1",
        objective_configuration_hash="obj1",
        code_commit="c1",
    )
    for field in REQUIRED_RESULT_HASH_FIELDS:
        assert field in hashes
        assert hashes[field]



@pytest.mark.unit
def test_phase_gate__after_holdout_start__configuration_mutation_is_rejected() -> None:
    """REPRO-003: after holdout start, frozen config mutation is rejected."""
    from epistemic_sycophancy.config.frozen import (
        ConfigImmutabilityError,
        freeze_experiment_config,
        mark_holdout_started,
    )
    from epistemic_sycophancy.reproducibility.artifacts import (
        REQUIRED_RESULT_HASH_FIELDS,
    )

    hashes = {name: f"hash-{name}" for name in REQUIRED_RESULT_HASH_FIELDS}
    frozen = freeze_experiment_config(
        config_payload={"tau": 1.0, "lambda_n": 0.1},
        **hashes,
    )
    assert frozen.freeze_status == "sealed"
    assert frozen.holdout_started is False
    started = mark_holdout_started(frozen)
    assert started.holdout_started is True
    with pytest.raises(ConfigImmutabilityError):
        started.replace_field("tau", 2.0)
