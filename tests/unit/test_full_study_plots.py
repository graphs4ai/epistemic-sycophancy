"""ORCH-PLOT-003/004: full_study validation figures (DEC-103)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_full_study_plots__ordered_labels__non_intervened_then_criteria() -> None:
    """ORCH-PLOT-003: label order is non_intervened then _LOSS_CRITERIA."""
    from epistemic_sycophancy.logging.full_study_plots import ordered_behavioral_labels

    series = {
        "l_total": 0.4,
        "l_resist": 0.5,
        "non_intervened": 0.3,
        "l_behavior": 0.35,
    }
    assert ordered_behavioral_labels(series) == [
        "non_intervened",
        "l_resist",
        "l_behavior",
        "l_total",
    ]


@pytest.mark.unit
def test_full_study_plots__metric_bars__writes_png_and_skips_empty(
    tmp_path: Path,
) -> None:
    """ORCH-PLOT-003: bar chart from non_intervened + criteria; empty → None."""
    from epistemic_sycophancy.logging.full_study_plots import (
        plot_behavioral_metric_bars,
    )

    series = {
        "non_intervened": 0.3,
        "l_resist": 0.5,
        "l_total": 0.4,
    }
    out = tmp_path / "metric_ftw.png"
    result = plot_behavioral_metric_bars(
        series,
        metric="ftw",
        order_regime="CF",
        output_path=out,
    )
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0

    empty = tmp_path / "empty.png"
    assert (
        plot_behavioral_metric_bars(
            {},
            metric="ftw",
            order_regime="CF",
            output_path=empty,
        )
        is None
    )
    assert not empty.exists()


def _margin_row(
    *,
    question_id: str,
    condition: str,
    partition: str,
    baseline: float,
    intervened: float,
) -> dict[str, object]:
    raw = intervened - baseline
    return {
        "question_id": question_id,
        "condition": condition,
        "partition": partition,
        "baseline_margin": baseline,
        "intervened_margin": intervened,
        "raw_delta": raw,
        "favorable_delta": raw,
        "baseline_truthful": baseline > 1e-6,
        "intervened_truthful": intervened > 1e-6,
    }


@pytest.mark.unit
def test_full_study_plots__ib_mean_favorable_delta__writes_png(
    tmp_path: Path,
) -> None:
    """ORCH-PLOT-004: grouped bars of mean IB favorable_delta by partition."""
    from epistemic_sycophancy.logging.full_study_plots import (
        plot_ib_mean_favorable_delta,
    )

    margins_by_criterion = {
        "l_resist": [
            _margin_row(
                question_id="q1",
                condition="IB",
                partition="q_plus",
                baseline=1.0,
                intervened=1.5,
            ),
            _margin_row(
                question_id="q2",
                condition="IB",
                partition="q_minus",
                baseline=-1.0,
                intervened=-0.2,
            ),
            _margin_row(
                question_id="q1",
                condition="N",
                partition="q_plus",
                baseline=1.0,
                intervened=1.0,
            ),
        ],
        "l_total": [
            _margin_row(
                question_id="q1",
                condition="IB",
                partition="q_plus",
                baseline=1.0,
                intervened=1.2,
            ),
            _margin_row(
                question_id="q2",
                condition="IB",
                partition="q_minus",
                baseline=-1.0,
                intervened=0.0,
            ),
        ],
    }
    out = tmp_path / "margins_ib_mean_favorable_delta.png"
    result = plot_ib_mean_favorable_delta(margins_by_criterion, output_path=out)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 0

    empty = tmp_path / "empty_ib.png"
    assert plot_ib_mean_favorable_delta({}, output_path=empty) is None
    assert not empty.exists()


@pytest.mark.unit
def test_full_study_plots__l_total_scatter_and_hist__write_pngs(
    tmp_path: Path,
) -> None:
    """ORCH-PLOT-004: l_total scatter + favorable_delta hist; empty → None."""
    from epistemic_sycophancy.logging.full_study_plots import (
        plot_margins_delta_hist_l_total,
        plot_margins_scatter_l_total,
    )

    rows = [
        _margin_row(
            question_id="q1",
            condition="N",
            partition="q_plus",
            baseline=1.0,
            intervened=1.1,
        ),
        _margin_row(
            question_id="q2",
            condition="IB",
            partition="q_minus",
            baseline=-0.5,
            intervened=0.2,
        ),
        _margin_row(
            question_id="q1",
            condition="CB",
            partition="q_plus",
            baseline=0.8,
            intervened=0.7,
        ),
        _margin_row(
            question_id="q2",
            condition="IB",
            partition="q_plus",
            baseline=2.0,
            intervened=2.1,
        ),
    ]
    scatter = tmp_path / "margins_scatter_baseline_vs_intervened_l_total.png"
    hist = tmp_path / "margins_favorable_delta_hist_l_total.png"
    assert plot_margins_scatter_l_total(rows, output_path=scatter) == scatter
    assert plot_margins_delta_hist_l_total(rows, output_path=hist) == hist
    assert scatter.is_file() and scatter.stat().st_size > 0
    assert hist.is_file() and hist.stat().st_size > 0

    assert plot_margins_scatter_l_total([], output_path=tmp_path / "s.png") is None
    assert plot_margins_delta_hist_l_total([], output_path=tmp_path / "h.png") is None
