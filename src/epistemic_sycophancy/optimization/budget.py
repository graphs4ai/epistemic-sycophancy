"""Matched optimizer budget accounting (Phase H OPT-011 / DEC-034)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BudgetCounters:
    """Declared forward/backward-equivalent budget counters."""

    n_objective_evals: int = 0
    n_forward_equiv: int = 0
    n_backward_equiv: int = 0
    n_tokens: int = 0
    wall_time_s: float | None = None
    gpu_time_s: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def record_cmaes_objective_eval(
    counters: BudgetCounters,
    *,
    n_tokens: int = 0,
) -> None:
    """CMA-ES: one full-corpus objective eval → +1 objective, +1 forward, +0 backward."""
    counters.n_objective_evals += 1
    counters.n_forward_equiv += 1
    counters.n_tokens += int(n_tokens)


def record_adam_full_split_step(
    counters: BudgetCounters,
    *,
    n_tokens: int = 0,
) -> None:
    """Adam: one full-split grad step → +1 objective-eq, +1 forward, +1 backward."""
    counters.n_objective_evals += 1
    counters.n_forward_equiv += 1
    counters.n_backward_equiv += 1
    counters.n_tokens += int(n_tokens)


def budgets_match(
    left: BudgetCounters,
    right: BudgetCounters,
    *,
    budget_match_on: str,
) -> bool:
    """True when the declared match counter is equal on both sides."""
    allowed = {"n_objective_evals", "n_forward_equiv", "n_backward_equiv"}
    if budget_match_on not in allowed:
        raise ValueError(
            f"budget_match_on must be one of {sorted(allowed)}; got {budget_match_on!r}"
        )
    return getattr(left, budget_match_on) == getattr(right, budget_match_on)
