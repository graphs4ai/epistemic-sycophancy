"""Cross-order evaluation matrix tests (Phase I ORDER-X)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.evaluation.cross_order import build_cross_order_matrix


ORDER_REGIMES = ("CF", "IF", "RO")


@pytest.mark.unit
def test_cross_order__selected_interventions__produce_nine_evaluation_cells() -> None:
    """ORDER-X-001: optimized_under × evaluated_under yields exactly nine cells."""
    betas = {
        "CF": [-1.0, -0.5],
        "IF": [-0.8, -0.2],
        "RO": [-0.3, 0.0],
    }
    opt_hashes = {"CF": "opt_cf", "IF": "opt_if", "RO": "opt_ro"}
    eval_hashes = {"CF": "eval_cf", "IF": "eval_if", "RO": "eval_ro"}
    partition_fps = {"CF": "part_cf", "IF": "part_if", "RO": "part_ro"}

    cells = build_cross_order_matrix(
        betas_by_optimized_under=betas,
        optimization_order_manifest_hashes=opt_hashes,
        evaluation_order_manifest_hashes=eval_hashes,
        baseline_partition_fingerprints=partition_fps,
        # Stub metrics keyed by evaluated_under (no model needed for matrix shape).
        metrics_by_evaluated_under={
            order: {
                "ftw": 0.1,
                "cbr": 0.2,
                "selectivity": 0.1,
                "n_q_plus": 2,
                "n_q_minus": 1,
            }
            for order in ORDER_REGIMES
        },
    )
    assert len(cells) == 9
    pairs = {(c.optimized_under, c.evaluated_under) for c in cells}
    expected = {(o, e) for o in ORDER_REGIMES for e in ORDER_REGIMES}
    assert pairs == expected


@pytest.mark.unit
def test_cross_order__beta_vector__is_not_refit_during_evaluation() -> None:
    """ORDER-X-002: cross-order eval copies β; does not mutate or refit it."""
    betas = {
        "CF": [-1.0, -0.5],
        "IF": [-0.8, -0.2],
        "RO": [-0.3, 0.0],
    }
    original = {k: list(v) for k, v in betas.items()}
    cells = build_cross_order_matrix(
        betas_by_optimized_under=betas,
        optimization_order_manifest_hashes={"CF": "a", "IF": "b", "RO": "c"},
        evaluation_order_manifest_hashes={"CF": "d", "IF": "e", "RO": "f"},
        baseline_partition_fingerprints={"CF": "p1", "IF": "p2", "RO": "p3"},
        metrics_by_evaluated_under={
            order: {
                "ftw": 0.0,
                "cbr": 0.0,
                "selectivity": 0.0,
                "n_q_plus": 1,
                "n_q_minus": 1,
            }
            for order in ORDER_REGIMES
        },
    )
    # Inputs unchanged after evaluation.
    assert betas == original
    # Cell β equals the opt-order vector and is a copy (tuple).
    for cell in cells:
        assert cell.beta == tuple(original[cell.optimized_under])
        assert isinstance(cell.beta, tuple)


@pytest.mark.unit
def test_cross_order__prompt_candidates__follow_evaluated_under_regime() -> None:
    """ORDER-X-003: candidate labeling follows evaluated_under, not optimized_under."""
    from epistemic_sycophancy.evaluation.cross_order import (
        resolve_evaluation_order_assignment,
    )
    from epistemic_sycophancy.prompts.ordering import assign_order

    # Opt under CF, eval under IF → truthful_label must be B (IF), not A (CF).
    assignment = resolve_evaluation_order_assignment(
        optimized_under="CF",
        evaluated_under="IF",
        question_id="q_cross",
        truthful_text="truth",
        incorrect_text="false",
        ro_seed=0,
    )
    expected = assign_order(
        order_regime="IF",
        truthful_text="truth",
        incorrect_text="false",
        question_id="q_cross",
        ro_seed=0,
    )
    assert assignment.truthful_label == expected.truthful_label
    assert assignment.truthful_label == "B"
    assert assignment.order_regime == "IF"
    # Must not silently use optimized_under CF labeling.
    cf = assign_order(
        order_regime="CF",
        truthful_text="truth",
        incorrect_text="false",
        question_id="q_cross",
        ro_seed=0,
    )
    assert assignment.truthful_label != cf.truthful_label


@pytest.mark.unit
def test_cross_order__matrix_cell__uses_evaluation_order_baseline_partition() -> None:
    """ORDER-X-004: matrix cell fingerprint matches evaluation-order partition.

    Opt under CF, eval under IF → use Q+_IF / Q-_IF fingerprint (BASE-004).
    """
    from epistemic_sycophancy.metrics.baseline_partition import (
        build_baseline_partition,
        freeze_baseline_partition_artifact,
        select_partition_for_evaluation,
    )

    partition_cf = build_baseline_partition(
        order_regime="CF",
        neutral_margins={"q1": 1.0, "q2": -1.0},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    partition_if = build_baseline_partition(
        order_regime="IF",
        neutral_margins={"q1": -1.0, "q2": 1.0},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    art_cf = freeze_baseline_partition_artifact(
        partition=partition_cf,
        model_revision_hash="m",
        prompt_template_hash="p",
        order_manifest_hash="cf",
        dataset_manifest_hash="d",
    )
    art_if = freeze_baseline_partition_artifact(
        partition=partition_if,
        model_revision_hash="m",
        prompt_template_hash="p",
        order_manifest_hash="if",
        dataset_manifest_hash="d",
    )
    selected = select_partition_for_evaluation(
        partitions_by_order={"CF": partition_cf, "IF": partition_if, "RO": partition_cf},
        optimization_order="CF",
        evaluation_order="IF",
    )
    assert selected is partition_if

    cells = build_cross_order_matrix(
        betas_by_optimized_under={
            "CF": [-1.0],
            "IF": [-0.5],
            "RO": [0.0],
        },
        optimization_order_manifest_hashes={"CF": "ocf", "IF": "oif", "RO": "oro"},
        evaluation_order_manifest_hashes={"CF": "ecf", "IF": "eif", "RO": "ero"},
        baseline_partition_fingerprints={
            "CF": art_cf.fingerprint,
            "IF": art_if.fingerprint,
            "RO": art_cf.fingerprint,
        },
        metrics_by_evaluated_under={
            order: {
                "ftw": 0.0,
                "cbr": 0.0,
                "selectivity": 0.0,
                "n_q_plus": 1,
                "n_q_minus": 1,
            }
            for order in ORDER_REGIMES
        },
    )
    cf_if = next(
        c for c in cells if c.optimized_under == "CF" and c.evaluated_under == "IF"
    )
    assert cf_if.baseline_partition_fingerprint == art_if.fingerprint
    assert cf_if.baseline_partition_fingerprint != art_cf.fingerprint


@pytest.mark.unit
def test_cross_order__cell_record__contains_optimization_and_evaluation_manifest_hashes() -> None:
    """ORDER-X-005: each cell stores opt and eval order-manifest hashes."""
    cells = build_cross_order_matrix(
        betas_by_optimized_under={"CF": [-1.0], "IF": [-0.5], "RO": [0.0]},
        optimization_order_manifest_hashes={
            "CF": "hash_opt_cf",
            "IF": "hash_opt_if",
            "RO": "hash_opt_ro",
        },
        evaluation_order_manifest_hashes={
            "CF": "hash_eval_cf",
            "IF": "hash_eval_if",
            "RO": "hash_eval_ro",
        },
        baseline_partition_fingerprints={"CF": "p_cf", "IF": "p_if", "RO": "p_ro"},
        metrics_by_evaluated_under={
            order: {
                "ftw": 0.0,
                "cbr": 0.0,
                "selectivity": 0.0,
                "n_q_plus": 1,
                "n_q_minus": 1,
            }
            for order in ORDER_REGIMES
        },
    )
    cell = next(
        c for c in cells if c.optimized_under == "IF" and c.evaluated_under == "RO"
    )
    assert cell.optimization_order_manifest_hash == "hash_opt_if"
    assert cell.evaluation_order_manifest_hash == "hash_eval_ro"
    assert cell.optimization_order_manifest_hash != cell.evaluation_order_manifest_hash
