"""Validation-only model selection tests (Phase H OPT-010)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_model_selection__candidate_choice__cannot_read_holdout_metrics() -> None:
    """OPT-010: selection API cannot access holdout metrics (DEC-033)."""
    from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
    from epistemic_sycophancy.optimization.selection import (
        ValidationMetricsCandidate,
        select_best_checkpoint,
    )

    candidates = [
        ValidationMetricsCandidate(
            checkpoint_id="ck_a",
            trial_index=0,
            l_total=1.5,
            selectivity=0.2,
        ),
        ValidationMetricsCandidate(
            checkpoint_id="ck_b",
            trial_index=1,
            l_total=1.1,
            selectivity=0.1,
        ),
    ]
    best = select_best_checkpoint(candidates)
    assert best.checkpoint_id == "ck_b"

    poisoned = {
        "checkpoint_id": "ck_holdout",
        "trial_index": 2,
        "l_total": 0.01,
        "holdout_l_total": 0.0,
    }
    with pytest.raises(HoldoutAccessError):
        select_best_checkpoint([poisoned])  # type: ignore[list-item]
