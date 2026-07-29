"""Tiny optimizer smoke stage (Phase K/L RUN-012 / WIRE-009)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.objective.total import evaluate_objective
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows
from epistemic_sycophancy.reproducibility.phase_gates import require_identity_gate


@dataclass(frozen=True)
class OptSmokeResult:
    """Finite objective smoke result on a tiny non-holdout subset."""

    l_total: float
    split_name: str
    holdout_accessed: bool
    question_ids: tuple[str, ...]


_ALLOWED_SPLITS = frozenset({"optimization", "feature_selection"})


def run_opt_smoke(
    *,
    question_ids: Sequence[str],
    split_name: str,
    beta: Sequence[float],
    freeze_status: str,
    identity_passed: bool,
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
    ib_margins_by_question: Mapping[object, Sequence[float]],
    cb_margins_by_question: Mapping[object, Sequence[float]],
    baseline_cb_margins: Mapping[object, Sequence[float]],
    baseline_neutral_margins: Mapping[object, float],
    current_neutral_margins: Mapping[object, float],
    q_plus: Sequence[object],
    q_minus: Sequence[object],
) -> OptSmokeResult:
    """Evaluate real ``evaluate_objective`` on a tiny non-holdout subset (DEC-062)."""
    require_identity_gate(identity_passed=identity_passed)
    if split_name.startswith("holdout") or split_name == "holdout_test_behavior":
        load_holdout_rows(freeze_status=freeze_status)
        raise HoldoutAccessError(f"opt smoke cannot use split {split_name!r}")
    if split_name not in _ALLOWED_SPLITS:
        raise HoldoutAccessError(
            f"opt smoke allows only {sorted(_ALLOWED_SPLITS)}; got {split_name!r}"
        )
    result = evaluate_objective(
        ib_margins_by_question=ib_margins_by_question,
        cb_margins_by_question=cb_margins_by_question,
        baseline_cb_margins=baseline_cb_margins,
        baseline_neutral_margins=baseline_neutral_margins,
        current_neutral_margins=current_neutral_margins,
        q_plus=q_plus,
        q_minus=q_minus,
        beta=beta,
        tau=tau,
        w_r=w_r,
        w_u=w_u,
        delta_n=delta_n,
        delta_c=delta_c,
        lambda_n=lambda_n,
        lambda_c=lambda_c,
        lambda_beta=lambda_beta,
    )
    return OptSmokeResult(
        l_total=float(result.l_total),
        split_name=split_name,
        holdout_accessed=False,
        question_ids=tuple(question_ids),
    )


def run_opt_smoke_adam_step(
    *,
    beta_init: Sequence[float],
    grad: Sequence[float],
    adam_lr: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    adam_microbatch_questions: int,
    beta_lower: float,
    beta_upper: float,
    max_steps: int,
) -> tuple[float, ...]:
    """Run ``max_steps`` ProjectedAdam updates from synthetic grads (WIRE-010)."""
    import torch

    from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1; got {max_steps!r}")
    beta = torch.nn.Parameter(
        torch.tensor(list(beta_init), dtype=torch.float64)
    )
    opt = ProjectedAdam(
        beta=beta,
        adam_lr=adam_lr,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        adam_microbatch_questions=adam_microbatch_questions,
        beta_lower=beta_lower,
        beta_upper=beta_upper,
    )
    grad_tensor = torch.tensor(list(grad), dtype=torch.float64)
    for _ in range(max_steps):
        opt.zero_grad()
        beta.grad = grad_tensor.clone()
        opt.step()
    return tuple(float(x) for x in beta.detach().tolist())
