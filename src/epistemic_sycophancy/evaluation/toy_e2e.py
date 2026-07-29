"""Toy end-to-end evaluation pipeline (Phase J / DEC-046)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from epistemic_sycophancy.metrics.baseline_partition import (
    BaselinePartition,
    build_baseline_partition,
    freeze_baseline_partition_artifact,
)
from epistemic_sycophancy.metrics.behavioral import (
    BehavioralMetrics,
    compute_behavioral_metrics,
)
from epistemic_sycophancy.scoring.margins import truthful_margin

# DEC-016 / DEC-046 identity head: score_A = r[0], score_B = r[1].
_HEAD = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

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
    """Unintervened toy baseline outputs for one order regime."""

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


def _score_row(row: ToyPromptRow) -> tuple[float, float, float]:
    residual = torch.tensor(row.residual_last, dtype=torch.float64)
    logits = _HEAD @ residual
    score_a = float(logits[0].item())
    score_b = float(logits[1].item())
    margin = truthful_margin(
        score_a=score_a, score_b=score_b, truthful_label=row.truthful_label
    )
    return score_a, score_b, margin


def run_toy_e2e_baseline(*, order_regime: str) -> ToyE2EBaselineResult:
    """Score the DEC-046 toy corpus without intervention for ``order_regime``."""
    rows = [row for row in build_dec046_corpus() if row.order_regime == order_regime]
    if not rows:
        raise ValueError(f"no DEC-046 rows for order_regime={order_regime!r}")

    logits_by_prompt_id: dict[str, tuple[float, float]] = {}
    margins_by_prompt_id: dict[str, float] = {}
    neutral_margins: dict[str, float] = {}
    ib_margins: dict[str, list[float]] = {}
    cb_margins: dict[str, list[float]] = {}

    for row in rows:
        score_a, score_b, margin = _score_row(row)
        logits_by_prompt_id[row.prompt_id] = (score_a, score_b)
        margins_by_prompt_id[row.prompt_id] = margin
        if row.condition == "N":
            neutral_margins[row.question_id] = margin
        elif row.condition == "IB":
            ib_margins.setdefault(row.question_id, []).append(margin)
        elif row.condition == "CB":
            cb_margins.setdefault(row.question_id, []).append(margin)

    partition = build_baseline_partition(
        order_regime=order_regime,
        neutral_margins=neutral_margins,
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
        current_neutral_margins=neutral_margins,
        current_ib_margins=ib_margins,
        current_cb_margins=cb_margins,
        epsilon=EPSILON,
    )
    return ToyE2EBaselineResult(
        logits_by_prompt_id=logits_by_prompt_id,
        margins_by_prompt_id=margins_by_prompt_id,
        neutral_margins=neutral_margins,
        ib_margins=ib_margins,
        cb_margins=cb_margins,
        partition=partition,
        metrics=metrics,
    )
