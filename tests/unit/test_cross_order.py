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
