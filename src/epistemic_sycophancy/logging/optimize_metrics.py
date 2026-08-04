"""Optimize metrics CSV + iteration curves (ORCH-LOG-CSV / ORCH-PLOT-002 / DEC-097)."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ITERATION_CSV_COLUMNS: tuple[str, ...] = (
    "index",
    "optimizer_kind",
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
    "number_at_lower_bound",
    "number_at_upper_bound",
)

STEP_CSV_COLUMNS: tuple[str, ...] = (
    *ITERATION_CSV_COLUMNS,
    "step_grad_norm",
)

ITERATION_PLOT_METRICS: tuple[str, ...] = (
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
    "number_at_lower_bound",
    "number_at_upper_bound",
)


def count_betas_at_bounds(
    beta: Sequence[float],
    *,
    beta_lower: float,
    beta_upper: float,
) -> tuple[int, int]:
    """Count coefficients exactly at lower / upper bounds after clamp."""
    lo = float(beta_lower)
    hi = float(beta_upper)
    n_lo = 0
    n_hi = 0
    for value in beta:
        v = float(value)
        if v == lo:
            n_lo += 1
        if v == hi:
            n_hi += 1
    return n_lo, n_hi


def write_optimize_metrics_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path | str,
    *,
    columns: Sequence[str],
) -> Path:
    """Write metric rows to CSV with the given column order."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    return out


def plot_iteration_metric_curves(
    rows: Sequence[Mapping[str, Any]],
    curves_dir: Path | str,
) -> dict[str, Path]:
    """Write one PNG per iteration metric; empty rows → no files."""
    if not rows:
        return {}

    directory = Path(curves_dir)
    directory.mkdir(parents=True, exist_ok=True)

    xs = [int(row["index"]) for row in rows]
    kind = rows[0].get("optimizer_kind")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: dict[str, Path] = {}
    for metric in ITERATION_PLOT_METRICS:
        ys: list[float] = []
        skip = False
        for row in rows:
            raw = row.get(metric, "")
            if raw is None or raw == "":
                skip = True
                break
            ys.append(float(raw))
        if skip:
            continue
        path = directory / f"{metric}.png"
        title = f"{metric} over iterations"
        if kind is not None:
            title = f"{metric} over iterations ({kind})"
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o", label=metric)
        ax.set_xlabel("index")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written[metric] = path
    return written
