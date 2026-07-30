"""Production objective_fn / grad_fn adapters (ORCH-024 / DEC-076)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.objective.total import (
    evaluate_objective,
    evaluate_objective_with_grad,
)
from epistemic_sycophancy.runner.adapters.margins import build_margin_payload


def build_objective_fn(
    study: StudyConfig,
    stack: Any,
    *,
    partitions: Mapping[str, Any],
    margin_scorer: Callable[..., Mapping[str, Any]] | None = None,
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
) -> Callable[[Sequence[float], Sequence[str]], Sequence[float]]:
    """Build ``(beta, eligible_qids) -> grad`` via local linearization at β.

    When ``margin_jacobian_fn`` is provided it must return
    ``ib_margin_jac`` / ``cb_margin_jac`` / ``neutral_margin_jac`` aligned to
    the live margin payload (GRAD-002 / DEC-084). Otherwise margin Jacobians
    default to zero (legacy stub until GRAD-004 wires the production builder).
    """

    def grad_fn(beta: Sequence[float], eligible_qids: Sequence[str]) -> Sequence[float]:
        payload = build_margin_payload(
            study,
            stack,
            beta=beta,
            question_ids=eligible_qids,
            partitions=partitions,
            margin_scorer=margin_scorer,
        )
        m = int(study.experiment.coefficient_length)
        zero_row = torch.zeros(m, dtype=torch.float64)

        def _as_seq(vals: Any) -> Sequence[Any]:
            if isinstance(vals, (list, tuple)):
                return vals
            return (vals,)

        def _zero_seq_map(src: Mapping[str, Any]) -> dict[str, list[torch.Tensor]]:
            return {
                str(qid): [zero_row.clone() for _ in _as_seq(vals)]
                for qid, vals in src.items()
            }

        if margin_jacobian_fn is not None:
            jac_payload = margin_jacobian_fn(
                beta=beta,
                question_ids=eligible_qids,
                partitions=partitions,
            )
            ib_margin_jac = jac_payload["ib_margin_jac"]
            cb_margin_jac = jac_payload["cb_margin_jac"]
            neutral_margin_jac = jac_payload["neutral_margin_jac"]
        else:
            ib_margin_jac = _zero_seq_map(payload["ib_margins_by_question"])
            cb_margin_jac = _zero_seq_map(payload["cb_margins_by_question"])
            neutral_margin_jac = {
                qid: zero_row.clone() for qid in payload["current_neutral_margins"]
            }

        exp = study.experiment
        beta_t = torch.tensor(list(beta), dtype=torch.float64)
        _loss, grad = evaluate_objective_with_grad(
            beta=beta_t,
            ib_margin_const=payload["ib_margins_by_question"],
            ib_margin_jac=ib_margin_jac,
            cb_margin_const=payload["cb_margins_by_question"],
            cb_margin_jac=cb_margin_jac,
            baseline_cb_margins=payload["baseline_cb_margins"],
            baseline_neutral_margins=payload["baseline_neutral_margins"],
            neutral_margin_const=payload["current_neutral_margins"],
            neutral_margin_jac=neutral_margin_jac,
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
        return tuple(float(x) for x in grad)

    return grad_fn
