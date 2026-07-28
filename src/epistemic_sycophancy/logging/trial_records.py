"""Objective component logging (Phase G OBJ-011 / DEC-026)."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_sycophancy.objective.total import ObjectiveResult

OBJECTIVE_VERSION_V1 = "v1_no_residual"


@dataclass(frozen=True)
class ObjectiveComponents:
    """Logged objective fields (DEC-026). Residual is not in the sum identity."""

    l_resist: float
    l_recover: float
    l_behavior: float
    l_neutral: float
    l_correct: float
    l_beta: float
    l_total: float
    l_residual_perturbation: float
    objective_version: str


def build_objective_components(
    result: ObjectiveResult,
    *,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
    l_residual_perturbation: float = 0.0,
    objective_version: str = OBJECTIVE_VERSION_V1,
) -> ObjectiveComponents:
    """Build logged components and verify the weighted sum identity."""
    components = ObjectiveComponents(
        l_resist=result.l_resist,
        l_recover=result.l_recover,
        l_behavior=result.l_behavior,
        l_neutral=result.l_neutral,
        l_correct=result.l_correct,
        l_beta=result.l_beta,
        l_total=result.l_total,
        l_residual_perturbation=float(l_residual_perturbation),
        objective_version=objective_version,
    )
    expected_total = (
        components.l_behavior
        + float(lambda_n) * components.l_neutral
        + float(lambda_c) * components.l_correct
        + float(lambda_beta) * components.l_beta
    )
    if abs(expected_total - components.l_total) > 1e-12 * max(
        1.0, abs(components.l_total)
    ):
        raise ValueError(
            "logged components do not sum to logged total under DEC-026 identity"
        )
    return components
