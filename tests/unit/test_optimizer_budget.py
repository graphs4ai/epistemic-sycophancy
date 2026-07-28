"""Optimizer budget accounting tests (Phase H OPT-011)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_optimizer_comparison__budget_accounting__uses_declared_forward_backward_equivalents() -> None:
    """OPT-011: budget counters use declared F/B equivalents (DEC-034)."""
    from epistemic_sycophancy.optimization.budget import (
        BudgetCounters,
        budgets_match,
        record_adam_full_split_step,
        record_cmaes_objective_eval,
    )

    cma = BudgetCounters()
    adam = BudgetCounters()

    record_cmaes_objective_eval(cma, n_tokens=10)
    record_cmaes_objective_eval(cma, n_tokens=10)
    record_adam_full_split_step(adam, n_tokens=20)
    record_adam_full_split_step(adam, n_tokens=20)

    assert cma.n_objective_evals == 2
    assert cma.n_forward_equiv == 2
    assert cma.n_backward_equiv == 0
    assert cma.n_tokens == 20
    assert set(cma.as_dict()) >= {
        "n_objective_evals",
        "n_forward_equiv",
        "n_backward_equiv",
        "n_tokens",
        "wall_time_s",
        "gpu_time_s",
    }

    assert adam.n_objective_evals == 2
    assert adam.n_forward_equiv == 2
    assert adam.n_backward_equiv == 2
    assert adam.n_tokens == 40

    assert budgets_match(cma, adam, budget_match_on="n_objective_evals")
    assert budgets_match(cma, adam, budget_match_on="n_forward_equiv")
    assert not budgets_match(cma, adam, budget_match_on="n_backward_equiv")
