"""Trial logging completeness tests (Phase H OPT-012)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_trial_logging__every_trial__contains_required_components_metrics_and_beta() -> None:
    """OPT-012: every trial record has required losses, metrics, and β (DEC-035)."""
    from epistemic_sycophancy.logging.trial_records import (
        ObjectiveComponents,
        TrialRecord,
        build_trial_record,
    )
    from epistemic_sycophancy.optimization.budget import BudgetCounters

    components = ObjectiveComponents(
        l_resist=0.7,
        l_recover=0.8,
        l_behavior=0.75,
        l_neutral=0.1,
        l_correct=0.2,
        l_beta=0.5,
        l_total=1.2,
        l_residual_perturbation=0.0,
        objective_version="v1_no_residual",
    )
    budget = BudgetCounters(
        n_objective_evals=1,
        n_forward_equiv=1,
        n_backward_equiv=0,
        n_tokens=12,
    )
    record = build_trial_record(
        components=components,
        beta=[-1.0, -0.5, 0.0],
        trial_index=0,
        optimizer_kind="cmaes",
        ro_manifest_hash="c" * 64,
        order_regime="RO",
        neutral_accuracy=0.5,
        ftw=0.25,
        cbr=0.5,
        selectivity=0.4,
        pra_mean=2.0 / 3.0,
        pra_all=0.0,
        n_questions_total=3,
        n_q_plus=2,
        n_q_minus=1,
        n_q_tie=0,
        n_ib_prompts=3,
        n_cb_prompts=4,
        n_invalid=0,
        budget=budget,
    )
    assert isinstance(record, TrialRecord)
    required = {
        "l_resist",
        "l_recover",
        "l_behavior",
        "l_neutral",
        "l_correct",
        "l_beta",
        "l_total",
        "beta",
        "neutral_accuracy",
        "ftw",
        "cbr",
        "selectivity",
        "pra_mean",
        "pra_all",
        "n_questions_total",
        "n_q_plus",
        "n_q_minus",
        "n_q_tie",
        "n_ib_prompts",
        "n_cb_prompts",
        "n_invalid",
        "trial_index",
        "optimizer_kind",
        "ro_manifest_hash",
        "order_regime",
        "n_objective_evals",
        "n_forward_equiv",
        "n_backward_equiv",
        "n_tokens",
        "wall_time_s",
        "gpu_time_s",
    }
    payload = record.as_dict()
    assert required <= set(payload)
    assert payload["beta"] == [-1.0, -0.5, 0.0]
    assert payload["l_total"] == 1.2

    with pytest.raises((TypeError, ValueError)):
        build_trial_record(
            components=components,
            beta=[-1.0],
            trial_index=1,
            optimizer_kind="cmaes",
            ro_manifest_hash="c" * 64,
            order_regime="RO",
        )
