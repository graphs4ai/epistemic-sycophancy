"""Toy end-to-end evaluation pipeline (Phase J / DEC-046)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta
from epistemic_sycophancy.metrics.baseline_partition import (
    BaselinePartition,
    build_baseline_partition,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import (
    BehavioralMetrics,
    compute_behavioral_metrics,
)
from epistemic_sycophancy.objective.total import ObjectiveResult, evaluate_objective
from epistemic_sycophancy.scoring.margins import truthful_margin

# DEC-016 / DEC-046 identity head: score_A = r[0], score_B = r[1].
_HEAD = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
_DECODER = torch.tensor(
    [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
    dtype=torch.float64,
)
_ENCODER = torch.tensor(
    [[0.5, 0.0], [0.0, 0.25], [0.25, 0.25]],
    dtype=torch.float64,
)
_ENCODER_BIAS = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)

# CF semantic margins matching §13.1 (DEC-046).
_CF_NEUTRAL: dict[str, float] = {"q1": 2.0, "q2": -1.0, "q3": 0.5}
_CF_IB: dict[str, list[float]] = {
    "q1": [1.0, -1.0],
    "q2": [-0.5, 0.5],
    "q3": [0.2],
}
_CF_CB: dict[str, list[float]] = {
    "q1": [2.2, 1.0],
    "q2": [2.0, -2.0, 1.0],
    "q3": [1.05],
}

EPSILON = 1e-6
TIE_POLICY = "merge_into_q_minus"
RO_TRUTHFUL_LABEL: dict[str, str] = {"q1": "A", "q2": "B", "q3": "A"}


@dataclass(frozen=True)
class ToyPromptRow:
    """One synthetic prompt row with a fixed last-token residual."""

    prompt_id: str
    question_id: str
    order_regime: str
    condition: str
    variant_index: int
    truthful_label: str
    residual_last: tuple[float, float]


@dataclass(frozen=True)
class ToyE2EBaselineResult:
    """Toy pipeline outputs for one order regime."""

    logits_by_prompt_id: dict[str, tuple[float, float]]
    margins_by_prompt_id: dict[str, float]
    neutral_margins: dict[str, float]
    ib_margins: dict[str, list[float]]
    cb_margins: dict[str, list[float]]
    partition: BaselinePartition
    metrics: BehavioralMetrics


def _row(
    *,
    order_regime: str,
    question_id: str,
    condition: str,
    variant_index: int,
    truthful_label: str,
    margin: float,
) -> ToyPromptRow:
    if truthful_label == "A":
        residual = (float(margin), 0.0)
    else:
        residual = (0.0, float(margin))
    return ToyPromptRow(
        prompt_id=f"{order_regime}:{question_id}:{condition}:{variant_index}",
        question_id=question_id,
        order_regime=order_regime,
        condition=condition,
        variant_index=variant_index,
        truthful_label=truthful_label,
        residual_last=residual,
    )


def build_dec046_corpus() -> tuple[ToyPromptRow, ...]:
    """Build the DEC-046 three-question CF/IF N/CB/IB corpus."""
    rows: list[ToyPromptRow] = []
    for order_regime, truthful_label in (("CF", "A"), ("IF", "B")):
        for qid, margin in _CF_NEUTRAL.items():
            rows.append(
                _row(
                    order_regime=order_regime,
                    question_id=qid,
                    condition="N",
                    variant_index=0,
                    truthful_label=truthful_label,
                    margin=margin,
                )
            )
        for qid, margins in _CF_IB.items():
            for idx, margin in enumerate(margins):
                rows.append(
                    _row(
                        order_regime=order_regime,
                        question_id=qid,
                        condition="IB",
                        variant_index=idx,
                        truthful_label=truthful_label,
                        margin=margin,
                    )
                )
        for qid, margins in _CF_CB.items():
            for idx, margin in enumerate(margins):
                rows.append(
                    _row(
                        order_regime=order_regime,
                        question_id=qid,
                        condition="CB",
                        variant_index=idx,
                        truthful_label=truthful_label,
                        margin=margin,
                    )
                )
    return tuple(rows)


def _score_residual(
    residual_last: tuple[float, float] | torch.Tensor,
    *,
    truthful_label: str,
) -> tuple[float, float, float]:
    if isinstance(residual_last, torch.Tensor):
        residual = residual_last.to(dtype=torch.float64)
    else:
        residual = torch.tensor(residual_last, dtype=torch.float64)
    logits = _HEAD @ residual
    score_a = float(logits[0].item())
    score_b = float(logits[1].item())
    margin = truthful_margin(
        score_a=score_a, score_b=score_b, truthful_label=truthful_label
    )
    return score_a, score_b, margin


def _score_row(row: ToyPromptRow) -> tuple[float, float, float]:
    return _score_residual(row.residual_last, truthful_label=row.truthful_label)


def _assemble_result(
    *,
    order_regime: str,
    rows: Sequence[ToyPromptRow],
    residuals: Sequence[torch.Tensor],
    baseline_neutral_margins: dict[str, float],
) -> ToyE2EBaselineResult:
    logits_by_prompt_id: dict[str, tuple[float, float]] = {}
    margins_by_prompt_id: dict[str, float] = {}
    current_neutral: dict[str, float] = {}
    ib_margins: dict[str, list[float]] = {}
    cb_margins: dict[str, list[float]] = {}

    for row, residual in zip(rows, residuals):
        score_a, score_b, margin = _score_residual(
            residual, truthful_label=row.truthful_label
        )
        logits_by_prompt_id[row.prompt_id] = (score_a, score_b)
        margins_by_prompt_id[row.prompt_id] = margin
        if row.condition == "N":
            current_neutral[row.question_id] = margin
        elif row.condition == "IB":
            ib_margins.setdefault(row.question_id, []).append(margin)
        elif row.condition == "CB":
            cb_margins.setdefault(row.question_id, []).append(margin)

    partition = build_baseline_partition(
        order_regime=order_regime,
        neutral_margins=baseline_neutral_margins,
        epsilon=EPSILON,
        tie_policy=TIE_POLICY,
    )
    artifact = freeze_baseline_partition_artifact(
        partition=partition,
        model_revision_hash="toy-e2e-dec046",
        prompt_template_hash="toy-e2e-dec046",
        order_manifest_hash=f"toy-e2e-{order_regime}",
        dataset_manifest_hash="toy-e2e-dec046",
    )
    metrics = compute_behavioral_metrics(
        frozen_partition=artifact,
        current_neutral_margins=current_neutral,
        current_ib_margins=ib_margins,
        current_cb_margins=cb_margins,
        epsilon=EPSILON,
    )
    return ToyE2EBaselineResult(
        logits_by_prompt_id=logits_by_prompt_id,
        margins_by_prompt_id=margins_by_prompt_id,
        neutral_margins=current_neutral,
        ib_margins=ib_margins,
        cb_margins=cb_margins,
        partition=partition,
        metrics=metrics,
    )


def run_toy_e2e_baseline(*, order_regime: str) -> ToyE2EBaselineResult:
    """Score the DEC-046 toy corpus without intervention for ``order_regime``."""
    rows = [row for row in build_dec046_corpus() if row.order_regime == order_regime]
    if not rows:
        raise ValueError(f"no DEC-046 rows for order_regime={order_regime!r}")
    residuals = [
        torch.tensor(row.residual_last, dtype=torch.float64) for row in rows
    ]
    baseline_neutral = {
        row.question_id: _score_row(row)[2] for row in rows if row.condition == "N"
    }
    return _assemble_result(
        order_regime=order_regime,
        rows=rows,
        residuals=residuals,
        baseline_neutral_margins=baseline_neutral,
    )


def run_toy_e2e_with_beta(
    *,
    order_regime: str,
    beta: Sequence[float],
    selected_indices: Sequence[int],
    scales: Sequence[float],
) -> ToyE2EBaselineResult:
    """Score the DEC-046 corpus with additive SAE intervention at ``beta``."""
    rows = [row for row in build_dec046_corpus() if row.order_regime == order_regime]
    if not rows:
        raise ValueError(f"no DEC-046 rows for order_regime={order_regime!r}")
    baseline_neutral = {
        row.question_id: _score_row(row)[2] for row in rows if row.condition == "N"
    }
    residuals: list[torch.Tensor] = []
    for row in rows:
        original = torch.tensor(row.residual_last, dtype=torch.float64)
        intervened = apply_additive_sae_delta(
            residual=original,
            selected_indices=list(selected_indices),
            scales=list(scales),
            beta=list(beta),
            encoder_weight=_ENCODER,
            encoder_bias=_ENCODER_BIAS,
            decoder_weight=_DECODER,
        )
        residuals.append(intervened)
    return _assemble_result(
        order_regime=order_regime,
        rows=rows,
        residuals=residuals,
        baseline_neutral_margins=baseline_neutral,
    )


@dataclass(frozen=True)
class ToyInterventionDetail:
    """Latents, delta, and logits for one intervened toy prompt."""

    latents: tuple[float, ...]
    latents_prime: tuple[float, ...]
    residual_delta: tuple[float, ...]
    logits: tuple[float, float]


def inspect_toy_e2e_prompt(
    *,
    prompt_id: str,
    beta: Sequence[float],
    selected_indices: Sequence[int],
    scales: Sequence[float],
) -> ToyInterventionDetail:
    """Return hand-checkable SAE intermediates for one DEC-046 prompt."""
    rows = {row.prompt_id: row for row in build_dec046_corpus()}
    try:
        row = rows[prompt_id]
    except KeyError as exc:
        raise KeyError(f"unknown DEC-046 prompt_id={prompt_id!r}") from exc
    residual = torch.tensor(row.residual_last, dtype=torch.float64)
    latents = torch.relu(residual @ _ENCODER.T + _ENCODER_BIAS)
    alphas = [float(s) * float(b) for s, b in zip(scales, beta)]
    latents_prime = latents.clone()
    for index, alpha in zip(selected_indices, alphas):
        latents_prime[index] = torch.relu(latents[index] + alpha)
    residual_delta = (latents_prime - latents) @ _DECODER
    intervened = residual + residual_delta
    logits = _HEAD @ intervened
    return ToyInterventionDetail(
        latents=tuple(float(v) for v in latents.tolist()),
        latents_prime=tuple(float(v) for v in latents_prime.tolist()),
        residual_delta=tuple(float(v) for v in residual_delta.tolist()),
        logits=(float(logits[0].item()), float(logits[1].item())),
    )


def evaluate_toy_e2e_objective(
    *,
    order_regime: str,
    beta: Sequence[float],
    selected_indices: Sequence[int],
    scales: Sequence[float],
    tau: float,
    w_r: float,
    w_u: float,
    delta_n: float,
    delta_c: float,
    lambda_n: float,
    lambda_c: float,
    lambda_beta: float,
) -> ObjectiveResult:
    """Evaluate the full objective on the intervened DEC-046 corpus."""
    scored = run_toy_e2e_with_beta(
        order_regime=order_regime,
        beta=beta,
        selected_indices=selected_indices,
        scales=scales,
    )
    baseline = run_toy_e2e_baseline(order_regime=order_regime)
    return evaluate_objective(
        ib_margins_by_question=scored.ib_margins,
        cb_margins_by_question=scored.cb_margins,
        baseline_cb_margins=baseline.cb_margins,
        baseline_neutral_margins=baseline.neutral_margins,
        current_neutral_margins=scored.neutral_margins,
        q_plus=baseline.partition.q_plus,
        q_minus=baseline.partition.q_minus,
        beta=list(beta),
        tau=tau,
        w_r=w_r,
        w_u=w_u,
        delta_n=delta_n,
        delta_c=delta_c,
        lambda_n=lambda_n,
        lambda_c=lambda_c,
        lambda_beta=lambda_beta,
    )


def toy_e2e_prompt_coefficient_jacobian(
    *,
    prompt_id: str,
    selected_indices: Sequence[int],
    scales: Sequence[float],
) -> list[float]:
    """Exact local ∂φ(M)/∂β on one DEC-046 prompt at β=0 (selected features)."""
    from epistemic_sycophancy.feature_selection.projected_gradient import (
        coefficient_jacobian,
        project_residual_gradient,
    )

    rows = {row.prompt_id: row for row in build_dec046_corpus()}
    try:
        row = rows[prompt_id]
    except KeyError as exc:
        raise KeyError(f"unknown DEC-046 prompt_id={prompt_id!r}") from exc
    residual = torch.tensor(row.residual_last, dtype=torch.float64, requires_grad=True)
    logits = _HEAD @ residual
    if row.truthful_label == "A":
        margin = logits[0] - logits[1]
    else:
        margin = logits[1] - logits[0]
    loss = torch.nn.functional.softplus(-margin)
    residual_grad = torch.autograd.grad(loss, residual)[0]
    raw = project_residual_gradient(gradient=residual_grad, decoder=_DECODER)
    latents = torch.relu(residual.detach() @ _ENCODER.T + _ENCODER_BIAS)
    full_scales = torch.ones(_DECODER.shape[0], dtype=torch.float64)
    for index, scale in zip(selected_indices, scales):
        full_scales[index] = float(scale)
    full_j = coefficient_jacobian(
        raw_projection=raw,
        latents=latents,
        feature_scales=full_scales,
    )
    return [float(full_j[index].item()) for index in selected_indices]
