"""Full-study validation figures (ORCH-PLOT-003/004 / DEC-103)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Mirror runner/full_study._LOSS_CRITERIA ordering for bar categories.
_LOSS_CRITERIA: tuple[str, ...] = (
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
)

BEHAVIORAL_PLOT_METRICS: tuple[str, ...] = (
    "neutral_accuracy",
    "ftw",
    "cbr",
    "selectivity",
    "pra_mean",
    "pra_all",
)


def ordered_behavioral_labels(series_by_label: Mapping[str, Any]) -> list[str]:
    """Return plot x-labels: ``non_intervened`` then present loss criteria."""
    labels: list[str] = []
    if "non_intervened" in series_by_label:
        labels.append("non_intervened")
    for criterion in _LOSS_CRITERIA:
        if criterion in series_by_label:
            labels.append(criterion)
    return labels


def plot_behavioral_metric_bars(
    series_by_label: Mapping[str, float],
    *,
    metric: str,
    order_regime: str,
    output_path: Path | str,
) -> Path | None:
    """Write a bar chart of one behavioral metric across checkpoints.

    Returns the output path, or ``None`` when ``series_by_label`` is empty.
    """
    labels = ordered_behavioral_labels(series_by_label)
    if not labels:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(series_by_label[label]) for label in labels]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(labels) + 2.0), 4.0))
    xs = list(range(len(labels)))
    ax.bar(xs, values, color="#4C72B0", edgecolor="black", linewidth=0.4)
    if "non_intervened" in series_by_label:
        ax.axhline(
            float(series_by_label["non_intervened"]),
            color="#C44E52",
            linestyle="--",
            linewidth=1.2,
            label="non_intervened",
        )
        ax.legend(loc="best")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by checkpoint ({order_regime})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _mean_ib_favorable_by_partition(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    sums: dict[str, float] = {"q_plus": 0.0, "q_minus": 0.0}
    counts: dict[str, int] = {"q_plus": 0, "q_minus": 0}
    for row in rows:
        if str(row.get("condition")) != "IB":
            continue
        partition = str(row.get("partition"))
        if partition not in sums:
            continue
        sums[partition] += float(row["favorable_delta"])
        counts[partition] += 1
    return {
        part: (sums[part] / counts[part] if counts[part] else float("nan"))
        for part in ("q_plus", "q_minus")
    }


def plot_ib_mean_favorable_delta(
    margins_by_criterion: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_path: Path | str,
) -> Path | None:
    """Grouped bars: mean IB favorable_delta by partition across criteria."""
    criteria = [c for c in _LOSS_CRITERIA if c in margins_by_criterion]
    if not criteria:
        return None

    means_plus: list[float] = []
    means_minus: list[float] = []
    for criterion in criteria:
        means = _mean_ib_favorable_by_partition(margins_by_criterion[criterion])
        means_plus.append(float(means["q_plus"]))
        means_minus.append(float(means["q_minus"]))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(criteria) + 2.0), 4.0))
    xs = np.arange(len(criteria), dtype=float)
    width = 0.38
    ax.bar(
        xs - width / 2,
        means_plus,
        width=width,
        label="q_plus (IB)",
        color="#55A868",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        xs + width / 2,
        means_minus,
        width=width,
        label="q_minus (IB)",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(criteria, rotation=35, ha="right")
    ax.set_ylabel("mean favorable_delta")
    ax.set_title("IB mean favorable margin delta by partition")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


_PARTITION_COLORS = {
    "q_plus": "#55A868",
    "q_minus": "#4C72B0",
    "q_tie": "#CCB974",
}


def plot_margins_scatter_l_total(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | str,
) -> Path | None:
    """Three-panel scatter: baseline vs intervened margins for N/IB/CB."""
    if not rows:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ("N", "IB", "CB")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharex=False, sharey=False)
    for ax, condition in zip(axes, conditions, strict=True):
        subset = [r for r in rows if str(r.get("condition")) == condition]
        for partition, color in _PARTITION_COLORS.items():
            pts = [r for r in subset if str(r.get("partition")) == partition]
            if not pts:
                continue
            xs = [float(r["baseline_margin"]) for r in pts]
            ys = [float(r["intervened_margin"]) for r in pts]
            ax.scatter(xs, ys, s=18, alpha=0.75, color=color, label=partition)
        all_vals = [
            float(r["baseline_margin"]) for r in subset
        ] + [float(r["intervened_margin"]) for r in subset]
        if all_vals:
            lo = min(all_vals)
            hi = max(all_vals)
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lim = (lo - pad, hi + pad)
            ax.plot(lim, lim, linestyle="--", color="gray", linewidth=1.0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_title(condition)
        ax.set_xlabel("baseline_margin")
        ax.set_ylabel("intervened_margin")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Baseline vs intervened margins (l_total)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_margins_delta_hist_l_total(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | str,
) -> Path | None:
    """Three-panel histogram of favorable_delta by condition for l_total."""
    if not rows:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ("N", "IB", "CB")
    all_deltas = [float(r["favorable_delta"]) for r in rows]
    if not all_deltas:
        return None
    lo = min(all_deltas)
    hi = max(all_deltas)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    bins = np.linspace(lo, hi, 21)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, condition in zip(axes, conditions, strict=True):
        deltas = [
            float(r["favorable_delta"])
            for r in rows
            if str(r.get("condition")) == condition
        ]
        ax.hist(deltas, bins=bins, color="#4C72B0", edgecolor="black", linewidth=0.3)
        ax.axvline(0.0, color="#C44E52", linestyle="--", linewidth=1.0)
        ax.set_title(condition)
        ax.set_xlabel("favorable_delta")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Favorable margin delta distribution (l_total)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_subset_mean_favorable_delta_l_total(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | str,
) -> Path | None:
    """Grouped bars: mean favorable_delta by baseline success/fail (DEC-104).

    Resistance = IB on q_plus; recovery = CB on q_minus. Empty → None.
    """
    from epistemic_sycophancy.analysis.margin_subsets import summarize_margin_subsets

    if not rows:
        return None
    summary = summarize_margin_subsets(rows)
    labels = [
        "resist_fail",
        "resist_ok",
        "recover_fail",
        "recover_ok",
    ]
    buckets = [
        summary["resistance"]["baseline_failing"],
        summary["resistance"]["baseline_successful"],
        summary["recovery"]["baseline_failing"],
        summary["recovery"]["baseline_successful"],
    ]
    if all(int(b["n"]) == 0 for b in buckets):
        return None
    values = [
        float(b["mean_favorable_delta"]) if b["mean_favorable_delta"] is not None else 0.0
        for b in buckets
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = list(range(len(labels)))
    ax.bar(xs, values, color="#55A868", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean favorable_delta")
    ax.set_title("l_total mean ΔM by baseline success/fail")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    for x, bucket in zip(xs, buckets, strict=True):
        ax.text(x, values[x], f"n={bucket['n']}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_context_contrast_delta_l_total(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | str,
) -> Path | None:
    """Grouped bars: mean delta_D_R / delta_D_U by partition (DEC-104)."""
    from epistemic_sycophancy.analysis.context_contrast import (
        build_context_contrast_rows,
        summarize_context_contrast,
    )

    if not rows:
        return None
    summary = summarize_context_contrast(build_context_contrast_rows(rows))
    partitions = ("q_plus", "q_minus")
    if all(int(summary[p]["n"]) == 0 for p in partitions):
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(partitions)
    d_r = [
        float(summary[p]["mean_delta_d_r"])
        if summary[p]["mean_delta_d_r"] is not None
        else 0.0
        for p in labels
    ]
    d_u = [
        float(summary[p]["mean_delta_d_u"])
        if summary[p]["mean_delta_d_u"] is not None
        else 0.0
        for p in labels
    ]
    xs = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(xs - width / 2, d_r, width, label="mean ΔD_R", color="#4C72B0")
    ax.bar(xs + width / 2, d_u, width, label="mean ΔD_U", color="#C44E52")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean delta contrast")
    ax.set_title("l_total context-contrast ΔD by partition")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_full_study_figures(
    *,
    behavioral_by_label: Mapping[str, Mapping[str, Any]],
    margins_by_criterion: Mapping[str, Sequence[Mapping[str, Any]]],
    out_dir: Path | str,
    order_regime: str,
) -> dict[str, str]:
    """Write the DEC-103 figure set; return artifact key → path for written files."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    for metric in BEHAVIORAL_PLOT_METRICS:
        series: dict[str, float] = {}
        for label, payload in behavioral_by_label.items():
            if metric not in payload:
                continue
            raw = payload[metric]
            if raw is None:
                continue
            series[label] = float(raw)
        path = plot_behavioral_metric_bars(
            series,
            metric=metric,
            order_regime=order_regime,
            output_path=directory / f"metric_{metric}.png",
        )
        if path is not None:
            artifacts[f"figure_metric_{metric}"] = str(path)

    ib_path = plot_ib_mean_favorable_delta(
        margins_by_criterion,
        output_path=directory / "margins_ib_mean_favorable_delta.png",
    )
    if ib_path is not None:
        artifacts["figure_margins_ib_mean_favorable_delta"] = str(ib_path)

    l_total_rows = margins_by_criterion.get("l_total")
    if l_total_rows:
        scatter = plot_margins_scatter_l_total(
            l_total_rows,
            output_path=directory
            / "margins_scatter_baseline_vs_intervened_l_total.png",
        )
        if scatter is not None:
            artifacts["figure_margins_scatter_l_total"] = str(scatter)
        hist = plot_margins_delta_hist_l_total(
            l_total_rows,
            output_path=directory / "margins_favorable_delta_hist_l_total.png",
        )
        if hist is not None:
            artifacts["figure_margins_delta_hist_l_total"] = str(hist)
        subset = plot_subset_mean_favorable_delta_l_total(
            l_total_rows,
            output_path=directory
            / "margins_subset_mean_favorable_delta_l_total.png",
        )
        if subset is not None:
            artifacts["figure_margins_subset_mean_favorable_delta_l_total"] = str(
                subset
            )
        contrast = plot_context_contrast_delta_l_total(
            l_total_rows,
            output_path=directory / "margins_context_contrast_delta_l_total.png",
        )
        if contrast is not None:
            artifacts["figure_margins_context_contrast_delta_l_total"] = str(contrast)

    return artifacts
