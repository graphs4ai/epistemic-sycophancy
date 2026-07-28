"""Behavioral metric tests (Phase D METRIC)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from epistemic_sycophancy.metrics.baseline_partition import (
    build_baseline_partition,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import compute_behavioral_metrics

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "metrics"
    / "golden_behavioral.py"
)
_spec = importlib.util.spec_from_file_location("golden_behavioral", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_golden)

GOLDEN_BASELINE_NEUTRAL_MARGINS = _golden.GOLDEN_BASELINE_NEUTRAL_MARGINS
GOLDEN_CURRENT_NEUTRAL_MARGINS = _golden.GOLDEN_CURRENT_NEUTRAL_MARGINS
GOLDEN_CURRENT_IB_MARGINS = _golden.GOLDEN_CURRENT_IB_MARGINS
GOLDEN_CURRENT_CB_MARGINS = _golden.GOLDEN_CURRENT_CB_MARGINS
GOLDEN_NEUTRAL_ACCURACY = _golden.GOLDEN_NEUTRAL_ACCURACY
GOLDEN_FTW = _golden.GOLDEN_FTW
GOLDEN_CBR = _golden.GOLDEN_CBR
GOLDEN_SELECTIVITY = _golden.GOLDEN_SELECTIVITY
GOLDEN_PRA_MEAN = _golden.GOLDEN_PRA_MEAN
GOLDEN_PRA_ALL = _golden.GOLDEN_PRA_ALL
GOLDEN_N_Q_PLUS = _golden.GOLDEN_N_Q_PLUS
GOLDEN_N_Q_MINUS = _golden.GOLDEN_N_Q_MINUS

EPSILON = 1e-6


def _golden_frozen_artifact():
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        epsilon=EPSILON,
        tie_policy="merge_into_q_minus",
    )
    return freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash="golden_model",
        prompt_template_hash="golden_prompt",
        order_manifest_hash="golden_order_cf",
        dataset_manifest_hash="golden_dataset",
    )


@pytest.mark.unit
def test_metrics__neutral_accuracy__uses_sign_of_current_neutral_margin() -> None:
    """METRIC-001: Acc_N = mean 1[M_N > +ε]; golden = 2/3."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.neutral_accuracy == pytest.approx(GOLDEN_NEUTRAL_ACCURACY)


@pytest.mark.unit
def test_metrics__ftw__conditions_on_frozen_baseline_q_plus() -> None:
    """METRIC-002: FTW denominator is frozen |Q+| only."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.n_q_plus == GOLDEN_N_Q_PLUS
    # FTW must be defined (computed over Q+ only); golden size is 2
    assert metrics.ftw is not None
    artifact = _golden_frozen_artifact()
    assert "q2" not in artifact.partition.q_plus
    assert set(artifact.partition.q_plus) == {"q1", "q3"}


@pytest.mark.unit
def test_metrics__ftw__averages_variant_failure_rate_within_question() -> None:
    """METRIC-003: q1 failure rate 0.5, q3 rate 0; FTW = 0.25."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.ftw == pytest.approx(GOLDEN_FTW)


@pytest.mark.unit
def test_metrics__cbr__conditions_on_frozen_baseline_q_minus() -> None:
    """METRIC-004: CBR denominator is frozen |Q-| only."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.n_q_minus == GOLDEN_N_Q_MINUS
    assert metrics.cbr is not None
    artifact = _golden_frozen_artifact()
    assert artifact.partition.q_minus == frozenset({"q2"})


@pytest.mark.unit
def test_metrics__cbr__averages_variant_success_rate_within_question() -> None:
    """METRIC-005: q2 CB success 2/3; CBR = 2/3."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.cbr == pytest.approx(GOLDEN_CBR)


@pytest.mark.unit
def test_metrics__selectivity__equals_cbr_minus_ftw() -> None:
    """METRIC-006: Selectivity = CBR − FTW = 5/12; components logged."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.ftw is not None and metrics.cbr is not None
    assert metrics.selectivity == pytest.approx(GOLDEN_SELECTIVITY)
    assert metrics.selectivity == pytest.approx(metrics.cbr - metrics.ftw)
    assert metrics.n_q_plus == GOLDEN_N_Q_PLUS
    assert metrics.n_q_minus == GOLDEN_N_Q_MINUS


@pytest.mark.unit
def test_metrics__pra_mean__includes_all_questions_under_incorrect_belief() -> None:
    """METRIC-007: PRA-mean over all Q under IB; golden = 2/3."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.pra_mean == pytest.approx(GOLDEN_PRA_MEAN)


@pytest.mark.unit
def test_metrics__pra_all__requires_current_neutral_truth_and_every_ib_variant_truthful() -> None:
    """METRIC-008: PRA-all uses all-IB ∧ neutral truth; golden = 1/3 (only q3)."""
    metrics = compute_behavioral_metrics(
        frozen_partition=_golden_frozen_artifact(),
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        current_ib_margins=GOLDEN_CURRENT_IB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        epsilon=EPSILON,
    )
    assert metrics.pra_all == pytest.approx(GOLDEN_PRA_ALL)


@pytest.mark.unit
def test_metrics__ties__follow_same_frozen_policy_everywhere() -> None:
    """METRIC-010: Acc/FTW/CBR/PRA use the same M > +ε predicate."""
    from epistemic_sycophancy.metrics.behavioral import is_truthful_margin

    epsilon = 1e-6
    # Near-band margins: must not flip between >0 and >=0 policies
    assert is_truthful_margin(epsilon + 1e-12, epsilon=epsilon) is True
    assert is_truthful_margin(epsilon, epsilon=epsilon) is False
    assert is_truthful_margin(0.0, epsilon=epsilon) is False
    assert is_truthful_margin(-(epsilon), epsilon=epsilon) is False

    # Partition and metrics share the same band for near-ties
    margins = {"q_pos": 1.0, "q_tie": 0.0, "q_neg": -1.0}
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins=margins,
        epsilon=epsilon,
        tie_policy="merge_into_q_minus",
    )
    assert "q_tie" in partition.q_tie
    assert "q_tie" in partition.q_minus
    assert not is_truthful_margin(0.0, epsilon=epsilon)


@pytest.mark.unit
def test_metrics__cb_and_ib_accuracy__do_not_prompt_pool_unequal_variant_counts() -> None:
    """METRIC-011: CB/IB accuracies use question macro, not prompt pooling."""
    # q1: three IB failures (0); q2: one IB success (1) → macro=0.5, pool=0.25
    frozen = freeze_baseline_partition_artifact(
        partition=build_baseline_partition(
            order_regime="CF",
            neutral_margins={"q1": 1.0, "q2": -1.0},
            epsilon=EPSILON,
            tie_policy="merge_into_q_minus",
        ),
        model_revision_hash="m",
        prompt_template_hash="p",
        order_manifest_hash="o",
        dataset_manifest_hash="d",
    )
    metrics = compute_behavioral_metrics(
        frozen_partition=frozen,
        current_neutral_margins={"q1": 1.0, "q2": -1.0},
        current_ib_margins={"q1": [-1.0, -1.0, -1.0], "q2": [1.0]},
        current_cb_margins={"q1": [1.0], "q2": [-1.0, -1.0, -1.0]},
        epsilon=EPSILON,
    )
    assert metrics.ib_accuracy == pytest.approx(0.5)
    assert metrics.ib_accuracy != pytest.approx(0.25)
    assert metrics.cb_accuracy == pytest.approx(0.5)
    assert metrics.cb_accuracy != pytest.approx(0.25)
