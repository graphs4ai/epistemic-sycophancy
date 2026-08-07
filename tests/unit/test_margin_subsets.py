"""ORCH-DIAG-001: success/fail margin subset summaries from validation JSONL rows."""

from __future__ import annotations

import pytest


def _row(
    *,
    question_id: str,
    condition: str,
    partition: str,
    baseline_margin: float,
    intervened_margin: float,
    baseline_truthful: bool,
    intervened_truthful: bool,
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
        "baseline_truthful": baseline_truthful,
        "intervened_truthful": intervened_truthful,
    }


@pytest.mark.unit
def test_margin_subsets__resistance_and_recovery__split_by_baseline_truthful() -> None:
    """ORCH-DIAG-001 / DEC-104: IB on Q+ and CB on Q- split by baseline_truthful.

    Hand-derived:
    Resistance (IB, q_plus):
      fail r1: M0=-2.5, Δ=+0.02 → favorable; no flip
      fail r2: M0=-1.0, Δ=-0.05 → adverse; no flip
      ok   r3: M0=+1.0, Δ=+0.10 → favorable; stays truthful
    Recovery (CB, q_minus):
      fail u1: M0=-0.5, Δ=+0.75 → favorable; flips to truthful
      ok   u2: M0=+2.0, Δ=0.0 → zero; stays truthful
    Neutral / other partitions ignored.
    """
    from epistemic_sycophancy.analysis.margin_subsets import summarize_margin_subsets

    rows = [
        _row(
            question_id="r1",
            condition="IB",
            partition="q_plus",
            baseline_margin=-2.5,
            intervened_margin=-2.48,
            baseline_truthful=False,
            intervened_truthful=False,
        ),
        _row(
            question_id="r2",
            condition="IB",
            partition="q_plus",
            baseline_margin=-1.0,
            intervened_margin=-1.05,
            baseline_truthful=False,
            intervened_truthful=False,
        ),
        _row(
            question_id="r3",
            condition="IB",
            partition="q_plus",
            baseline_margin=1.0,
            intervened_margin=1.1,
            baseline_truthful=True,
            intervened_truthful=True,
        ),
        _row(
            question_id="u1",
            condition="CB",
            partition="q_minus",
            baseline_margin=-0.5,
            intervened_margin=0.25,
            baseline_truthful=False,
            intervened_truthful=True,
        ),
        _row(
            question_id="u2",
            condition="CB",
            partition="q_minus",
            baseline_margin=2.0,
            intervened_margin=2.0,
            baseline_truthful=True,
            intervened_truthful=True,
        ),
        # Ignored: wrong condition / partition.
        _row(
            question_id="noise",
            condition="N",
            partition="q_plus",
            baseline_margin=3.0,
            intervened_margin=3.1,
            baseline_truthful=True,
            intervened_truthful=True,
        ),
        _row(
            question_id="noise2",
            condition="IB",
            partition="q_minus",
            baseline_margin=-3.0,
            intervened_margin=-2.0,
            baseline_truthful=False,
            intervened_truthful=False,
        ),
    ]

    summary = summarize_margin_subsets(rows)

    resist_fail = summary["resistance"]["baseline_failing"]
    assert resist_fail["n"] == 2
    assert resist_fail["mean_baseline_margin"] == pytest.approx((-2.5 + -1.0) / 2)
    assert resist_fail["median_baseline_margin"] == pytest.approx((-2.5 + -1.0) / 2)
    assert resist_fail["mean_favorable_delta"] == pytest.approx((0.02 + -0.05) / 2)
    assert resist_fail["median_favorable_delta"] == pytest.approx((0.02 + -0.05) / 2)
    assert resist_fail["n_favorable"] == 1
    assert resist_fail["n_adverse"] == 1
    assert resist_fail["n_zero"] == 0
    assert resist_fail["n_flip_to_truthful"] == 0
    assert resist_fail["n_flip_to_untruthful"] == 0

    resist_ok = summary["resistance"]["baseline_successful"]
    assert resist_ok["n"] == 1
    assert resist_ok["mean_baseline_margin"] == pytest.approx(1.0)
    assert resist_ok["mean_favorable_delta"] == pytest.approx(0.1)
    assert resist_ok["n_favorable"] == 1
    assert resist_ok["n_adverse"] == 0
    assert resist_ok["n_zero"] == 0

    recover_fail = summary["recovery"]["baseline_failing"]
    assert recover_fail["n"] == 1
    assert recover_fail["mean_baseline_margin"] == pytest.approx(-0.5)
    assert recover_fail["mean_favorable_delta"] == pytest.approx(0.75)
    assert recover_fail["n_favorable"] == 1
    assert recover_fail["n_flip_to_truthful"] == 1
    assert recover_fail["n_flip_to_untruthful"] == 0

    recover_ok = summary["recovery"]["baseline_successful"]
    assert recover_ok["n"] == 1
    assert recover_ok["mean_favorable_delta"] == pytest.approx(0.0)
    assert recover_ok["n_zero"] == 1
    assert recover_ok["n_favorable"] == 0
    assert recover_ok["n_adverse"] == 0
