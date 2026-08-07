"""ORCH-DIAG-002: paired context-contrast D_R / D_U from validation JSONL rows."""

from __future__ import annotations

import pytest


def _row(
    *,
    question_id: str,
    condition: str,
    partition: str,
    baseline_margin: float,
    intervened_margin: float,
) -> dict[str, object]:
    delta = intervened_margin - baseline_margin
    return {
        "question_id": question_id,
        "condition": condition,
        "partition": partition,
        "baseline_margin": baseline_margin,
        "intervened_margin": intervened_margin,
        "raw_delta": delta,
        "favorable_delta": delta,
        "baseline_truthful": baseline_margin > 0.0,
        "intervened_truthful": intervened_margin > 0.0,
    }


@pytest.mark.unit
def test_context_contrast__paired_deltas__match_hand_derivation() -> None:
    """ORCH-DIAG-002 / DEC-104: D_R=M_IB−M_N, D_U=M_CB−M_N and ΔD.

    Hand-derived for q1 on q_plus:
      N:  M0=3.0 → 3.05 (ΔN=+0.05)
      IB: M0=-2.0 → -1.7 (ΔIB=+0.30)
      CB: M0=2.5 → 2.6 (ΔCB=+0.10)
      D_R0 = -2-3 = -5; D_R1 = -1.7-3.05 = -4.75; delta_D_R = +0.25
           (= ΔIB − ΔN = 0.30 − 0.05)
      D_U0 = 2.5-3 = -0.5; D_U1 = 2.6-3.05 = -0.45; delta_D_U = +0.05
    q2 on q_minus ignored for q_plus share stats but still emitted.
    """
    from epistemic_sycophancy.analysis.context_contrast import (
        build_context_contrast_rows,
        summarize_context_contrast,
    )

    rows = [
        _row(
            question_id="q1",
            condition="N",
            partition="q_plus",
            baseline_margin=3.0,
            intervened_margin=3.05,
        ),
        _row(
            question_id="q1",
            condition="IB",
            partition="q_plus",
            baseline_margin=-2.0,
            intervened_margin=-1.7,
        ),
        _row(
            question_id="q1",
            condition="CB",
            partition="q_plus",
            baseline_margin=2.5,
            intervened_margin=2.6,
        ),
        _row(
            question_id="q2",
            condition="N",
            partition="q_minus",
            baseline_margin=-1.0,
            intervened_margin=-0.9,
        ),
        _row(
            question_id="q2",
            condition="IB",
            partition="q_minus",
            baseline_margin=-1.5,
            intervened_margin=-1.4,
        ),
        _row(
            question_id="q2",
            condition="CB",
            partition="q_minus",
            baseline_margin=0.5,
            intervened_margin=0.8,
        ),
    ]

    contrast_rows = build_context_contrast_rows(rows)
    by_q = {r["question_id"]: r for r in contrast_rows}
    assert set(by_q) == {"q1", "q2"}

    q1 = by_q["q1"]
    assert q1["partition"] == "q_plus"
    assert q1["baseline_d_r"] == pytest.approx(-5.0)
    assert q1["intervened_d_r"] == pytest.approx(-4.75)
    assert q1["delta_d_r"] == pytest.approx(0.25)
    assert q1["baseline_d_u"] == pytest.approx(-0.5)
    assert q1["intervened_d_u"] == pytest.approx(-0.45)
    assert q1["delta_d_u"] == pytest.approx(0.05)
    assert q1["delta_m_n"] == pytest.approx(0.05)
    assert q1["delta_m_ib"] == pytest.approx(0.30)
    assert q1["delta_m_cb"] == pytest.approx(0.10)
    assert q1["neutral_beats_ib_favorable"] is False  # ΔIB > ΔN

    q2 = by_q["q2"]
    assert q2["partition"] == "q_minus"
    assert q2["delta_d_r"] == pytest.approx(0.0)  # (−1.4−−1.5) − (−0.9−−1.0) = 0
    assert q2["delta_d_u"] == pytest.approx(0.2)  # (0.8−0.5) − (−0.9−−1.0) = 0.2

    summary = summarize_context_contrast(contrast_rows)
    assert summary["q_plus"]["n"] == 1
    assert summary["q_plus"]["mean_delta_d_r"] == pytest.approx(0.25)
    assert summary["q_plus"]["mean_delta_d_u"] == pytest.approx(0.05)
    assert summary["q_plus"]["share_neutral_beats_ib_favorable"] == pytest.approx(0.0)
    assert summary["q_minus"]["n"] == 1
    assert summary["q_minus"]["mean_delta_d_r"] == pytest.approx(0.0)
    assert summary["q_minus"]["mean_delta_d_u"] == pytest.approx(0.2)
