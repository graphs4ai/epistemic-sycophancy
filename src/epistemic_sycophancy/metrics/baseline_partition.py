"""Order-specific frozen baseline partitions (Phase D)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaselinePartition:
    """Frozen order-specific baseline partition over original questions."""

    order_regime: str
    q_plus: frozenset[str]
    q_minus: frozenset[str]
    q_tie: frozenset[str] = field(default_factory=frozenset)
    n_q_tie: int = 0
    epsilon: float = 0.0
    tie_policy: str = "merge_into_q_minus"


def build_baseline_partition(
    *,
    order_regime: str,
    neutral_margins: Mapping[str, float],
    epsilon: float,
    tie_policy: str,
    belief_conditioned_margins: Mapping[str, float] | None = None,
    intervened_margins: Mapping[str, float] | None = None,
) -> BaselinePartition:
    """Build an order-specific partition from unmodified neutral margins.

    Band assignment (BASE-005 / DEC-001):
        M > +ε → Q+
        M < -ε → Q-
        otherwise → Q_tie

    With ``tie_policy="merge_into_q_minus"``, Q_tie is merged into Q- for
    denominators; ``n_q_tie`` reports the pre-merge count.

    ``belief_conditioned_margins`` and ``intervened_margins`` are accepted for
    API clarity but must never affect the partition (BASE-002).
    """
    del belief_conditioned_margins, intervened_margins  # never used (BASE-002)

    if tie_policy != "merge_into_q_minus":
        raise ValueError(f"unsupported tie_policy: {tie_policy!r}")

    q_plus: set[str] = set()
    q_minus: set[str] = set()
    q_tie: set[str] = set()
    for question_id, margin in neutral_margins.items():
        m = float(margin)
        if m > epsilon:
            q_plus.add(question_id)
        elif m < -epsilon:
            q_minus.add(question_id)
        else:
            q_tie.add(question_id)

    n_q_tie = len(q_tie)
    # DEC-001: merge Q_tie into Q-
    q_minus_merged = q_minus | q_tie
    return BaselinePartition(
        order_regime=order_regime,
        q_plus=frozenset(q_plus),
        q_minus=frozenset(q_minus_merged),
        q_tie=frozenset(q_tie),
        n_q_tie=n_q_tie,
        epsilon=float(epsilon),
        tie_policy=tie_policy,
    )


def select_partition_for_evaluation(
    *,
    partitions_by_order: Mapping[str, BaselinePartition],
    optimization_order: str,
    evaluation_order: str,
) -> BaselinePartition:
    """Return the baseline partition for the evaluation order (BASE-004).

    ``optimization_order`` is accepted for call-site clarity but must not
    select the partition; only ``evaluation_order`` determines denominators.
    """
    del optimization_order  # never used for selection (BASE-004)
    try:
        return partitions_by_order[evaluation_order]
    except KeyError as exc:
        raise KeyError(
            f"no baseline partition for evaluation_order={evaluation_order!r}"
        ) from exc
