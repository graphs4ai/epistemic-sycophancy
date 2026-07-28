"""CMA-ES optimizer tests (Phase H OPT-002+)."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_optimizer_objective__cmaes_trial__evaluates_every_eligible_optimization_row() -> None:
    """OPT-002: each CMA-ES trial evaluates every eligible optimization question."""
    from epistemic_sycophancy.optimization.cmaes import CMAESOptimizer

    eligible = ("q1", "q2", "q3")
    seen_calls: list[frozenset[str]] = []

    def evaluate_on_questions(beta: list[float], question_ids: list[str]) -> float:
        seen_calls.append(frozenset(question_ids))
        return float(sum(abs(v) for v in beta))

    optimizer = CMAESOptimizer(
        x0=[0.0, 0.0, 0.0],
        sigma0=0.5,
        cma_seed=7,
        beta_lower=-2.0,
        beta_upper=0.0,
        eligible_question_ids=eligible,
    )
    candidates = optimizer.ask()
    assert len(candidates) >= 1
    for beta in candidates:
        loss = optimizer.evaluate_candidate(beta, evaluate_on_questions=evaluate_on_questions)
        assert isinstance(loss, float)

    assert seen_calls, "expected at least one corpus evaluation"
    for seen in seen_calls:
        assert seen == frozenset(eligible)
