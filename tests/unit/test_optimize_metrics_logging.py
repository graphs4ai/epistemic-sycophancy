"""ORCH-LOG-CSV-001 / ORCH-PLOT-002: optimize metrics CSV + iteration plots (DEC-097)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


@pytest.mark.unit
def test_count_betas_at_bounds__exact_equality__counts_lower_and_upper() -> None:
    """ORCH-LOG-CSV-001a: bound counts use exact equality on clamped values."""
    from epistemic_sycophancy.logging.optimize_metrics import count_betas_at_bounds

    n_lo, n_hi = count_betas_at_bounds(
        (-2.0, -1.0, 0.0, -2.0, -0.5),
        beta_lower=-2.0,
        beta_upper=0.0,
    )
    assert n_lo == 2
    assert n_hi == 1


@pytest.mark.unit
def test_write_optimize_metrics_csv__rows__write_requested_columns(
    tmp_path: Path,
) -> None:
    """ORCH-LOG-CSV-001b: CSV writes only the requested columns in order."""
    from epistemic_sycophancy.logging.optimize_metrics import (
        ITERATION_CSV_COLUMNS,
        STEP_CSV_COLUMNS,
        write_optimize_metrics_csv,
    )

    rows = [
        {
            "index": 0,
            "optimizer_kind": "projected_adam",
            "l_resist": 0.1,
            "l_recover": 0.2,
            "l_behavior": 0.15,
            "l_neutral": 0.0,
            "l_correct": 0.0,
            "l_beta": 0.01,
            "l_total": 0.16,
            "number_at_lower_bound": 1,
            "number_at_upper_bound": 0,
            "step_grad_norm": 1.5,
        },
        {
            "index": 1,
            "optimizer_kind": "projected_adam",
            "l_resist": 0.05,
            "l_recover": 0.1,
            "l_behavior": 0.075,
            "l_neutral": 0.0,
            "l_correct": 0.0,
            "l_beta": 0.02,
            "l_total": 0.095,
            "number_at_lower_bound": 2,
            "number_at_upper_bound": 0,
            "step_grad_norm": 0.9,
        },
    ]
    step_path = write_optimize_metrics_csv(
        rows, tmp_path / "steps.csv", columns=STEP_CSV_COLUMNS
    )
    assert step_path.is_file()
    with step_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(STEP_CSV_COLUMNS)
        step_rows = list(reader)
    assert len(step_rows) == 2
    assert step_rows[0]["step_grad_norm"] == "1.5"
    assert step_rows[1]["l_total"] == "0.095"

    iter_path = write_optimize_metrics_csv(
        rows, tmp_path / "iterations.csv", columns=ITERATION_CSV_COLUMNS
    )
    with iter_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(ITERATION_CSV_COLUMNS)
        assert "step_grad_norm" not in reader.fieldnames
        assert len(list(reader)) == 2


@pytest.mark.unit
def test_plot_iteration_metric_curves__writes_png_per_metric(
    tmp_path: Path,
) -> None:
    """ORCH-PLOT-002: non-empty iteration rows write one PNG per metric; empty → none."""
    from epistemic_sycophancy.logging.optimize_metrics import (
        ITERATION_PLOT_METRICS,
        plot_iteration_metric_curves,
    )

    rows = [
        {
            "index": 0,
            "l_resist": 0.5,
            "l_recover": 0.4,
            "l_behavior": 0.45,
            "l_neutral": 0.1,
            "l_correct": 0.2,
            "l_beta": 0.01,
            "l_total": 0.76,
            "number_at_lower_bound": 1,
            "number_at_upper_bound": 0,
            "optimizer_kind": "projected_adam",
        },
        {
            "index": 1,
            "l_resist": 0.4,
            "l_recover": 0.3,
            "l_behavior": 0.35,
            "l_neutral": 0.05,
            "l_correct": 0.1,
            "l_beta": 0.02,
            "l_total": 0.52,
            "number_at_lower_bound": 2,
            "number_at_upper_bound": 1,
            "optimizer_kind": "projected_adam",
        },
    ]
    curves_dir = tmp_path / "curves"
    written = plot_iteration_metric_curves(rows, curves_dir)
    assert set(written) == set(ITERATION_PLOT_METRICS)
    for metric, path in written.items():
        assert path == curves_dir / f"{metric}.png"
        assert path.is_file()
        assert path.stat().st_size > 0

    empty_dir = tmp_path / "empty_curves"
    empty_written = plot_iteration_metric_curves([], empty_dir)
    assert empty_written == {}
    assert not empty_dir.exists()
