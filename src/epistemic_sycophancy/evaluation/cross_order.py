"""Cross-order evaluation matrix (Phase I ORDER-X)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.prompts.ordering import OrderAssignment, assign_order


ORDER_REGIMES: tuple[str, ...] = ("CF", "IF", "RO")


@dataclass(frozen=True)
class CrossOrderCellRecord:
    """One optimized_under × evaluated_under cell (DEC-041)."""

    optimized_under: str
    evaluated_under: str
    beta: tuple[float, ...]
    optimization_order_manifest_hash: str
    evaluation_order_manifest_hash: str
    baseline_partition_fingerprint: str
    ftw: float
    cbr: float
    selectivity: float
    n_q_plus: int
    n_q_minus: int


def build_cross_order_matrix(
    *,
    betas_by_optimized_under: Mapping[str, Sequence[float]],
    optimization_order_manifest_hashes: Mapping[str, str],
    evaluation_order_manifest_hashes: Mapping[str, str],
    baseline_partition_fingerprints: Mapping[str, str],
    metrics_by_evaluated_under: Mapping[str, Mapping[str, float | int]],
) -> list[CrossOrderCellRecord]:
    """Build the 3×3 cross-order evaluation matrix (ORDER-X-001).

    β vectors are copied into each cell and never refit (ORDER-X-002).
    Prompts/partitions follow evaluated_under (ORDER-X-003/004).
    """
    cells: list[CrossOrderCellRecord] = []
    for optimized_under in ORDER_REGIMES:
        beta = tuple(float(x) for x in betas_by_optimized_under[optimized_under])
        for evaluated_under in ORDER_REGIMES:
            metrics = metrics_by_evaluated_under[evaluated_under]
            cells.append(
                CrossOrderCellRecord(
                    optimized_under=optimized_under,
                    evaluated_under=evaluated_under,
                    beta=beta,
                    optimization_order_manifest_hash=optimization_order_manifest_hashes[
                        optimized_under
                    ],
                    evaluation_order_manifest_hash=evaluation_order_manifest_hashes[
                        evaluated_under
                    ],
                    baseline_partition_fingerprint=baseline_partition_fingerprints[
                        evaluated_under
                    ],
                    ftw=float(metrics["ftw"]),
                    cbr=float(metrics["cbr"]),
                    selectivity=float(metrics["selectivity"]),
                    n_q_plus=int(metrics["n_q_plus"]),
                    n_q_minus=int(metrics["n_q_minus"]),
                )
            )
    return cells


def resolve_evaluation_order_assignment(
    *,
    optimized_under: str,
    evaluated_under: str,
    question_id: str,
    truthful_text: str,
    incorrect_text: str,
    ro_seed: int | None = None,
) -> OrderAssignment:
    """Resolve A/B labeling from evaluated_under only (ORDER-X-003).

    ``optimized_under`` is accepted for call-site clarity but must not select
    the prompt/candidate mapping.
    """
    del optimized_under
    return assign_order(
        order_regime=evaluated_under,
        truthful_text=truthful_text,
        incorrect_text=incorrect_text,
        question_id=question_id,
        ro_seed=ro_seed,
    )
