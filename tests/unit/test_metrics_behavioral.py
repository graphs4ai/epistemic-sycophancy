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
