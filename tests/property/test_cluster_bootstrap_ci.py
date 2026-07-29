"""Property tests for cluster-bootstrap percentile CIs (STAT-006)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.metrics.baseline_partition import build_baseline_partition
from epistemic_sycophancy.statistics.cluster_bootstrap import (
    bootstrap_selectivity_interval,
)


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=20, deadline=None)
def test_cluster_bootstrap__percentile_interval__has_ordered_finite_bounds(
    seed: int,
) -> None:
    """STAT-006: 95% percentile CI has finite, ordered bounds (L ≤ U)."""
    partition = build_baseline_partition(
        order_regime="CF",
        neutral_margins={"q1": 2.0, "q2": -1.0, "q3": 0.5},
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )
    result = bootstrap_selectivity_interval(
        frozen_partition=partition,
        current_neutral_margins={"q1": 1.4, "q2": -0.2, "q3": 0.8},
        current_ib_margins={"q1": [1.0, -1.0], "q2": [-0.5, 0.5], "q3": [0.2]},
        current_cb_margins={"q1": [2.2, 1.0], "q2": [2.0, -2.0, 1.0], "q3": [1.05]},
        epsilon=1e-6,
        n_replicates=20,
        seed=seed,
        bootstrap_ci_percentile=95.0,
    )
    assert math.isfinite(result.ci_low)
    assert math.isfinite(result.ci_high)
    assert result.ci_low <= result.ci_high
