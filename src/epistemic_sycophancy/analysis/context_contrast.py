"""Paired context-contrast diagnostics from validation margin JSONL (DEC-104)."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def build_context_contrast_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join N/IB/CB by question_id into D_R / D_U contrast rows.

    ``D_R = M_IB - M_N``, ``D_U = M_CB - M_N``.
    ``delta_D_* = D_*(intervened) - D_*(baseline)`` (= ``ΔM_cond - ΔM_N``).
    ``neutral_beats_ib_favorable`` is True when ``ΔM_N > ΔM_IB``.
    """
    by_qid: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        qid = str(row["question_id"])
        condition = str(row["condition"])
        by_qid.setdefault(qid, {})[condition] = row

    out: list[dict[str, Any]] = []
    for qid in sorted(by_qid):
        conds = by_qid[qid]
        missing = [c for c in ("N", "IB", "CB") if c not in conds]
        if missing:
            raise ValueError(
                f"context contrast requires N/IB/CB for question_id={qid!r}; "
                f"missing {missing}"
            )
        n_row = conds["N"]
        ib_row = conds["IB"]
        cb_row = conds["CB"]
        partition = str(n_row["partition"])
        if str(ib_row["partition"]) != partition or str(cb_row["partition"]) != partition:
            raise ValueError(
                f"partition mismatch for question_id={qid!r}: "
                f"N={partition!r}, IB={ib_row['partition']!r}, "
                f"CB={cb_row['partition']!r}"
            )

        m_n0 = float(n_row["baseline_margin"])
        m_n1 = float(n_row["intervened_margin"])
        m_ib0 = float(ib_row["baseline_margin"])
        m_ib1 = float(ib_row["intervened_margin"])
        m_cb0 = float(cb_row["baseline_margin"])
        m_cb1 = float(cb_row["intervened_margin"])

        delta_m_n = m_n1 - m_n0
        delta_m_ib = m_ib1 - m_ib0
        delta_m_cb = m_cb1 - m_cb0
        baseline_d_r = m_ib0 - m_n0
        intervened_d_r = m_ib1 - m_n1
        baseline_d_u = m_cb0 - m_n0
        intervened_d_u = m_cb1 - m_n1
        out.append(
            {
                "question_id": qid,
                "partition": partition,
                "baseline_d_r": baseline_d_r,
                "intervened_d_r": intervened_d_r,
                "delta_d_r": intervened_d_r - baseline_d_r,
                "baseline_d_u": baseline_d_u,
                "intervened_d_u": intervened_d_u,
                "delta_d_u": intervened_d_u - baseline_d_u,
                "delta_m_n": delta_m_n,
                "delta_m_ib": delta_m_ib,
                "delta_m_cb": delta_m_cb,
                "neutral_beats_ib_favorable": delta_m_n > delta_m_ib,
            }
        )
    return out


def _partition_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_delta_d_r": None,
            "mean_delta_d_u": None,
            "share_neutral_beats_ib_favorable": None,
        }
    delta_d_r = [float(r["delta_d_r"]) for r in rows]
    delta_d_u = [float(r["delta_d_u"]) for r in rows]
    n_beats = sum(1 for r in rows if bool(r["neutral_beats_ib_favorable"]))
    return {
        "n": len(rows),
        "mean_delta_d_r": float(statistics.fmean(delta_d_r)),
        "mean_delta_d_u": float(statistics.fmean(delta_d_u)),
        "share_neutral_beats_ib_favorable": float(n_beats) / float(len(rows)),
    }


def summarize_context_contrast(
    contrast_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate contrast rows by partition (q_plus / q_minus / q_tie)."""
    by_partition: dict[str, list[Mapping[str, Any]]] = {
        "q_plus": [],
        "q_minus": [],
        "q_tie": [],
    }
    for row in contrast_rows:
        partition = str(row["partition"])
        if partition not in by_partition:
            raise ValueError(f"unknown partition {partition!r}")
        by_partition[partition].append(row)
    return {
        name: _partition_summary(group) for name, group in by_partition.items()
    }
