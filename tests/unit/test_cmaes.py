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


@pytest.mark.unit
def test_cmaes__suggested_coefficients__respect_configured_bounds() -> None:
    """OPT-003: CMA-ES proposals stay within CFG-004 suppression-only bounds."""
    from epistemic_sycophancy.optimization.cmaes import CMAESOptimizer

    beta_lower = -2.0
    beta_upper = 0.0
    optimizer = CMAESOptimizer(
        x0=[0.0, 0.0, 0.0],
        sigma0=2.0,
        cma_seed=11,
        beta_lower=beta_lower,
        beta_upper=beta_upper,
        eligible_question_ids=("q1",),
    )
    for _ in range(5):
        candidates = optimizer.ask()
        for beta in candidates:
            assert len(beta) == 3
            for value in beta:
                assert beta_lower <= value <= beta_upper
        # feed tell so ask continues exploring
        values = [float(sum(v * v for v in beta)) for beta in candidates]
        optimizer.tell(candidates, values)


@pytest.mark.unit
def test_cmaes__repeated_trials__reuse_same_random_order_manifest() -> None:
    """OPT-004: repeated CMA-ES trials reuse the same frozen RO manifest."""
    from epistemic_sycophancy.optimization.cmaes import CMAESOptimizer
    from epistemic_sycophancy.prompts.ordering import build_ro_manifest, hash_ro_manifest

    question_ids = ["q1", "q2", "q3"]
    ro_manifest = build_ro_manifest(ro_seed=42, question_ids=question_ids)
    expected_hash = hash_ro_manifest(ro_manifest)

    optimizer = CMAESOptimizer(
        x0=[0.0, 0.0, 0.0],
        sigma0=0.5,
        cma_seed=3,
        beta_lower=-2.0,
        beta_upper=0.0,
        eligible_question_ids=question_ids,
        ro_manifest=ro_manifest,
    )

    hashes: list[str] = []
    for _ in range(3):
        candidates = optimizer.ask()
        hashes.append(optimizer.ro_manifest_hash)
        values = [float(sum(abs(v) for v in beta)) for beta in candidates]
        optimizer.tell(candidates, values)

    assert hashes == [expected_hash, expected_hash, expected_hash]
    assert optimizer.ro_manifest is ro_manifest
