"""REAL-006: real-model objective trial smoke."""

from __future__ import annotations

from dataclasses import fields

import pytest

from epistemic_sycophancy.evaluation.real_model_smoke import (
    real_model_objective_trial_smoke,
)
from epistemic_sycophancy.logging.trial_records import TrialRecord
from tests.real_model._pin import MODEL_ID, MODEL_REVISION


@pytest.mark.real_model
@pytest.mark.slow
def test_real_model__objective_trial_smoke__finite_logged_deterministic() -> None:
    """REAL-006: tiny objective smoke is finite, logged, and repeatable."""
    first = real_model_objective_trial_smoke(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        question_ids=("q1", "q2"),
        seed=0,
    )
    second = real_model_objective_trial_smoke(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        question_ids=("q1", "q2"),
        seed=0,
    )
    assert first.l_total == first.l_total  # finite (not NaN)
    assert abs(first.l_total) < float("inf")
    assert set(first.question_ids) == {"q1", "q2"}
    for field in fields(TrialRecord):
        value = getattr(first.trial_record, field.name)
        if field.name in {"wall_time_s", "gpu_time_s"}:
            continue
        assert value is not None
    assert first.l_total == second.l_total
    assert first.trial_record.l_total == second.trial_record.l_total
