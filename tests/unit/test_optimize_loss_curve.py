"""ORCH-PLOT-001: loss-over-trials plot from slim trial rows (DEC-091)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_optimize__loss_curve__writes_png_from_trial_rows(tmp_path: Path) -> None:
    """ORCH-PLOT-001: non-empty trials write a PNG; empty trials write nothing."""
    from epistemic_sycophancy.logging.loss_curve import plot_loss_over_trials

    out = tmp_path / "loss_curve.png"
    trials = [
        {"trial_index": 0, "l_total": 1.5, "optimizer_kind": "projected_adam"},
        {"trial_index": 1, "l_total": 1.2, "optimizer_kind": "projected_adam"},
        {"trial_index": 2, "l_total": 1.3, "optimizer_kind": "projected_adam"},
    ]
    result = plot_loss_over_trials(trials, out)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0

    empty_out = tmp_path / "empty_curve.png"
    empty_result = plot_loss_over_trials([], empty_out)
    assert empty_result is None
    assert not empty_out.exists()
