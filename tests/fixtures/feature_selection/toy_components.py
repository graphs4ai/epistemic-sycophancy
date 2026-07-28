"""Toy prompt rows and a frozen baseline partition for Phase F components."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_sycophancy.metrics.baseline_partition import (
    BaselinePartition,
    build_baseline_partition,
)


@dataclass(frozen=True)
class PromptRow:
    """One rendered prompt of the feature-selection split."""

    question_id: str
    condition: str  # "N" | "CB" | "IB"
    belief_variant_id: str | None
    margin: float


# Neutral baseline margins: q1 and q3 land in Q+, q2 in Q-.
NEUTRAL_BASELINE_MARGINS: dict[str, float] = {
    "q1": 2.0,
    "q2": -1.0,
    "q3": 0.5,
}


def frozen_partition(*, order_regime: str = "CF") -> BaselinePartition:
    """Frozen CF partition: Q+ = {q1, q3}, Q- = {q2} (DEC-013 epsilon)."""
    return build_baseline_partition(
        order_regime=order_regime,
        neutral_margins=NEUTRAL_BASELINE_MARGINS,
        epsilon=1e-6,
        tie_policy="merge_into_q_minus",
    )


def prompt_rows() -> tuple[PromptRow, ...]:
    """Rows with unequal belief-variant counts across the three questions."""
    return (
        PromptRow("q1", "N", None, 2.0),
        PromptRow("q1", "CB", "q1-cb-a", 2.2),
        PromptRow("q1", "CB", "q1-cb-b", 1.0),
        PromptRow("q1", "IB", "q1-ib-a", 1.0),
        PromptRow("q1", "IB", "q1-ib-b", -1.0),
        PromptRow("q2", "N", None, -1.0),
        PromptRow("q2", "CB", "q2-cb-a", 2.0),
        PromptRow("q2", "CB", "q2-cb-b", -2.0),
        PromptRow("q2", "CB", "q2-cb-c", 1.0),
        PromptRow("q2", "IB", "q2-ib-a", -0.5),
        PromptRow("q3", "N", None, 0.5),
        PromptRow("q3", "CB", "q3-cb-a", 1.05),
        PromptRow("q3", "IB", "q3-ib-a", 0.2),
    )
