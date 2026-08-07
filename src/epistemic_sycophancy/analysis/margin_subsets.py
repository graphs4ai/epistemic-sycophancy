"""Success/fail margin subset summaries from full_study validation JSONL (DEC-104)."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def _empty_bucket() -> dict[str, Any]:
    return {
        "n": 0,
        "mean_baseline_margin": None,
        "median_baseline_margin": None,
        "mean_favorable_delta": None,
        "median_favorable_delta": None,
        "n_favorable": 0,
        "n_adverse": 0,
        "n_zero": 0,
        "n_flip_to_truthful": 0,
        "n_flip_to_untruthful": 0,
    }


def _summarize_bucket(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one success/fail bucket; zero-delta uses exact float equality."""
    if not rows:
        return _empty_bucket()

    baselines = [float(r["baseline_margin"]) for r in rows]
    deltas = [float(r["favorable_delta"]) for r in rows]
    n_favorable = sum(1 for d in deltas if d > 0.0)
    n_adverse = sum(1 for d in deltas if d < 0.0)
    n_zero = sum(1 for d in deltas if d == 0.0)
    n_flip_to_truthful = sum(
        1
        for r in rows
        if (not bool(r["baseline_truthful"])) and bool(r["intervened_truthful"])
    )
    n_flip_to_untruthful = sum(
        1
        for r in rows
        if bool(r["baseline_truthful"]) and (not bool(r["intervened_truthful"]))
    )
    return {
        "n": len(rows),
        "mean_baseline_margin": float(statistics.fmean(baselines)),
        "median_baseline_margin": float(statistics.median(baselines)),
        "mean_favorable_delta": float(statistics.fmean(deltas)),
        "median_favorable_delta": float(statistics.median(deltas)),
        "n_favorable": n_favorable,
        "n_adverse": n_adverse,
        "n_zero": n_zero,
        "n_flip_to_truthful": n_flip_to_truthful,
        "n_flip_to_untruthful": n_flip_to_untruthful,
    }


def summarize_margin_subsets(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Split target rows by baseline_truthful (DEC-104).

    Resistance: condition IB on partition q_plus.
    Recovery: condition CB on partition q_minus.
    """
    resistance_fail: list[Mapping[str, Any]] = []
    resistance_ok: list[Mapping[str, Any]] = []
    recovery_fail: list[Mapping[str, Any]] = []
    recovery_ok: list[Mapping[str, Any]] = []

    for row in rows:
        condition = str(row["condition"])
        partition = str(row["partition"])
        if condition == "IB" and partition == "q_plus":
            if bool(row["baseline_truthful"]):
                resistance_ok.append(row)
            else:
                resistance_fail.append(row)
        elif condition == "CB" and partition == "q_minus":
            if bool(row["baseline_truthful"]):
                recovery_ok.append(row)
            else:
                recovery_fail.append(row)

    return {
        "resistance": {
            "baseline_failing": _summarize_bucket(resistance_fail),
            "baseline_successful": _summarize_bucket(resistance_ok),
        },
        "recovery": {
            "baseline_failing": _summarize_bucket(recovery_fail),
            "baseline_successful": _summarize_bucket(recovery_ok),
        },
    }
