"""Production objective_fn / grad_fn adapters (ORCH-024 / DEC-076 / DEC-084)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.objective.total import (
    evaluate_objective,
    evaluate_objective_with_local_grad,
)
from epistemic_sycophancy.runner.adapters.margin_jacobian import build_margin_jacobian_fn
from epistemic_sycophancy.runner.adapters.margins import (
    MarginBaselineCache,
    build_margin_payload,
)


def build_objective_fn(
    study: StudyConfig,
    stack: Any,
    *,
    partitions: Mapping[str, Any],
    margin_scorer: Callable[..., Mapping[str, Any]] | None = None,
    baseline_cache: MarginBaselineCache | None = None,
) -> Callable[[Sequence[float], Sequence[str]], float]:
    """Build ``(beta, eligible_qids) -> l_total`` with live margins (DEC-076)."""

    def objective_fn(beta: Sequence[float], eligible_qids: Sequence[str]) -> float:
        payload = build_margin_payload(
            study,
            stack,
            beta=beta,
            question_ids=eligible_qids,
            partitions=partitions,
            margin_scorer=margin_scorer,
            baseline_cache=baseline_cache,
        )
        exp = study.experiment
        result = evaluate_objective(
            ib_margins_by_question=payload["ib_margins_by_question"],
            cb_margins_by_question=payload["cb_margins_by_question"],
            baseline_cb_margins=payload["baseline_cb_margins"],
            baseline_neutral_margins=payload["baseline_neutral_margins"],
            current_neutral_margins=payload["current_neutral_margins"],
            q_plus=payload["q_plus"],
            q_minus=payload["q_minus"],
            beta=beta,
            tau=float(exp.tau),
            w_r=float(exp.w_r),
            w_u=float(exp.w_u),
            delta_n=float(exp.delta_n),
            delta_c=float(exp.delta_c),
            lambda_n=float(exp.lambda_n),
            lambda_c=float(exp.lambda_c),
            lambda_beta=float(exp.lambda_beta),
        )
        return float(result.l_total)

    return objective_fn


def build_grad_fn(
    study: StudyConfig,
    stack: Any,
    *,
    partitions: Mapping[str, Any],
    margin_scorer: Callable[..., Mapping[str, Any]] | None = None,
    margin_jacobian_fn: Callable[..., Mapping[str, Any]] | None = None,
    baseline_cache: MarginBaselineCache | None = None,
) -> Callable[[Sequence[float], Sequence[str]], Sequence[float]]:
    """Build ``(beta, eligible_qids) -> grad`` via local linearization at β.

    Live margins M(β₀) plus ∂M/∂β form M_local(δ)=M(β₀)+J·δ at δ=0 (DEC-095).
    Margin Jacobians come from ``margin_jacobian_fn`` when provided; otherwise
    from ``build_margin_jacobian_fn(study, stack)`` (projected ∂M/∂β, DEC-084).
    Identically zero or non-finite grads raise (DEC-084 loud-fail).
    """
    resolved_jac_fn: Callable[..., Mapping[str, Any]] | None
    if margin_jacobian_fn is not None:
        resolved_jac_fn = margin_jacobian_fn
    elif hasattr(stack, "margin_projection_batch"):
        resolved_jac_fn = build_margin_jacobian_fn(study, stack)
    else:
        resolved_jac_fn = None

    def grad_fn(beta: Sequence[float], eligible_qids: Sequence[str]) -> Sequence[float]:
        if resolved_jac_fn is None:
            raise ValueError(
                "build_grad_fn requires margin_jacobian_fn or "
                "stack.margin_projection_batch for projected ∂M/∂β (DEC-084); "
                "all-zero margin jac default is removed"
            )
        payload = build_margin_payload(
            study,
            stack,
            beta=beta,
            question_ids=eligible_qids,
            partitions=partitions,
            margin_scorer=margin_scorer,
            baseline_cache=baseline_cache,
        )
        m = int(study.experiment.coefficient_length)
        jac_payload = resolved_jac_fn(
            beta=beta,
            question_ids=eligible_qids,
            partitions=partitions,
        )
        exp = study.experiment
        beta_t = torch.tensor(list(beta), dtype=torch.float64)
        _loss, grad = evaluate_objective_with_local_grad(
            beta=beta_t,
            ib_margins_live=payload["ib_margins_by_question"],
            ib_margin_jac=jac_payload["ib_margin_jac"],
            cb_margins_live=payload["cb_margins_by_question"],
            cb_margin_jac=jac_payload["cb_margin_jac"],
            baseline_cb_margins=payload["baseline_cb_margins"],
            baseline_neutral_margins=payload["baseline_neutral_margins"],
            neutral_margins_live=payload["current_neutral_margins"],
            neutral_margin_jac=jac_payload["neutral_margin_jac"],
            q_plus=payload["q_plus"],
            q_minus=payload["q_minus"],
            tau=float(exp.tau),
            w_r=float(exp.w_r),
            w_u=float(exp.w_u),
            delta_n=float(exp.delta_n),
            delta_c=float(exp.delta_c),
            lambda_n=float(exp.lambda_n),
            lambda_c=float(exp.lambda_c),
            lambda_beta=float(exp.lambda_beta),
        )
        del _loss
        if len(grad) != m:
            raise ValueError(f"grad length {len(grad)} != coefficient_length {m}")
        if not all(math.isfinite(float(x)) for x in grad):
            raise ValueError(
                f"∂L/∂β is non-finite (DEC-084): grad={list(grad)!r}"
            )
        grad_norm_sq = sum(float(x) * float(x) for x in grad)
        if grad_norm_sq == 0.0:
            raise ValueError(
                "∂L/∂β is identically zero after projected margin Jacobians "
                "(DEC-084 loud-fail); refusing silent Adam no-op"
            )
        return tuple(float(x) for x in grad)

    return grad_fn
