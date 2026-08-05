"""Objective component and trial logging (Phase G OBJ-011 / Phase H OPT-012)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from epistemic_sycophancy.objective.total import ObjectiveResult

OBJECTIVE_VERSION_V1 = "v1_no_residual"
OBJECTIVE_VERSION_V2 = "v2_soft_hinge_no_residual"
# Current default: softplus preservation hinges (DEC-101); residual still excluded.
OBJECTIVE_VERSION_CURRENT = OBJECTIVE_VERSION_V2


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


@dataclass(frozen=True)
class TrialRecord:
    """Complete optimizer trial log (OPT-012 / DEC-035)."""

    l_resist: float
    l_recover: float
    l_behavior: float
    l_neutral: float
    l_correct: float
    l_beta: float
    l_total: float
    l_residual_perturbation: float
    objective_version: str
    beta: tuple[float, ...]
    neutral_accuracy: float
    ftw: float
    cbr: float
    selectivity: float
    pra_mean: float
    pra_all: float
    n_questions_total: int
    n_q_plus: int
    n_q_minus: int
    n_q_tie: int
    n_ib_prompts: int
    n_cb_prompts: int
    n_invalid: int
    trial_index: int
    optimizer_kind: str
    ro_manifest_hash: str
    order_regime: str
    n_objective_evals: int
    n_forward_equiv: int
    n_backward_equiv: int
    n_tokens: int
    wall_time_s: float | None
    gpu_time_s: float | None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["beta"] = list(self.beta)
        return payload


def build_objective_components(
    result: ObjectiveResult,
    *,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
    l_residual_perturbation: float = 0.0,
    objective_version: str = OBJECTIVE_VERSION_CURRENT,
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


def build_trial_record(
    *,
    components: ObjectiveComponents,
    beta: Sequence[float],
    trial_index: int,
    optimizer_kind: str,
    ro_manifest_hash: str,
    order_regime: str,
    neutral_accuracy: float,
    ftw: float,
    cbr: float,
    selectivity: float,
    pra_mean: float,
    pra_all: float,
    n_questions_total: int,
    n_q_plus: int,
    n_q_minus: int,
    n_q_tie: int,
    n_ib_prompts: int,
    n_cb_prompts: int,
    n_invalid: int,
    budget: object,
) -> TrialRecord:
    """Build a complete trial record; missing required kwargs fail at call time."""
    return TrialRecord(
        l_resist=components.l_resist,
        l_recover=components.l_recover,
        l_behavior=components.l_behavior,
        l_neutral=components.l_neutral,
        l_correct=components.l_correct,
        l_beta=components.l_beta,
        l_total=components.l_total,
        l_residual_perturbation=components.l_residual_perturbation,
        objective_version=components.objective_version,
        beta=tuple(float(v) for v in beta),
        neutral_accuracy=float(neutral_accuracy),
        ftw=float(ftw),
        cbr=float(cbr),
        selectivity=float(selectivity),
        pra_mean=float(pra_mean),
        pra_all=float(pra_all),
        n_questions_total=int(n_questions_total),
        n_q_plus=int(n_q_plus),
        n_q_minus=int(n_q_minus),
        n_q_tie=int(n_q_tie),
        n_ib_prompts=int(n_ib_prompts),
        n_cb_prompts=int(n_cb_prompts),
        n_invalid=int(n_invalid),
        trial_index=int(trial_index),
        optimizer_kind=str(optimizer_kind),
        ro_manifest_hash=str(ro_manifest_hash),
        order_regime=str(order_regime),
        n_objective_evals=int(budget.n_objective_evals),  # type: ignore[attr-defined]
        n_forward_equiv=int(budget.n_forward_equiv),  # type: ignore[attr-defined]
        n_backward_equiv=int(budget.n_backward_equiv),  # type: ignore[attr-defined]
        n_tokens=int(budget.n_tokens),  # type: ignore[attr-defined]
        wall_time_s=budget.wall_time_s,  # type: ignore[attr-defined]
        gpu_time_s=budget.gpu_time_s,  # type: ignore[attr-defined]
    )
