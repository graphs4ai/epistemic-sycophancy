"""Tiny optimizer smoke stage (Phase K RUN-012)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
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
) -> OptSmokeResult:
    """Evaluate a deterministic finite stand-in objective on a tiny subset."""
    require_identity_gate(identity_passed=identity_passed)
    if split_name.startswith("holdout") or split_name == "holdout_test_behavior":
        load_holdout_rows(freeze_status=freeze_status)
        raise HoldoutAccessError(f"opt smoke cannot use split {split_name!r}")
    if split_name not in _ALLOWED_SPLITS:
        raise HoldoutAccessError(
            f"opt smoke allows only {sorted(_ALLOWED_SPLITS)}; got {split_name!r}"
        )
    # Deterministic finite surrogate: sum of squares of beta (β-only).
    l_total = float(sum(float(b) * float(b) for b in beta))
    return OptSmokeResult(
        l_total=l_total,
        split_name=split_name,
        holdout_accessed=False,
        question_ids=tuple(question_ids),
    )
