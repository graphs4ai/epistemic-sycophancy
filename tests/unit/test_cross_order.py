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
