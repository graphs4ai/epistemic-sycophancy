"""CF / IF / RO answer-order assignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OrderAssignment:
    """Mapped A/B candidates and truthful/incorrect labels for one order regime."""

    candidate_a: str
    candidate_b: str
    truthful_label: str
    incorrect_label: str
    order_regime: str
    order_manifest_id: str | None = None


def assign_order(
    *,
    order_regime: str,
    truthful_text: str,
    incorrect_text: str,
    question_id: str | None = None,
    ro_seed: int | None = None,
) -> OrderAssignment:
    """Map truthful/incorrect texts onto A/B for the requested order regime."""
    if order_regime == "CF":
        return OrderAssignment(
            candidate_a=truthful_text,
            candidate_b=incorrect_text,
            truthful_label="A",
            incorrect_label="B",
            order_regime="CF",
        )
    raise ValueError(f"unsupported order_regime: {order_regime!r}")
