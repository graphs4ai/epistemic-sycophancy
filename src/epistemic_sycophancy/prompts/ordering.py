"""CF / IF / RO answer-order assignment."""

from __future__ import annotations

import hashlib
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


def _ro_truthful_label(*, ro_seed: int, question_id: str) -> str:
    """DEC-009: SHA-256(f\"{ro_seed}\\0{question_id}\"); LSB 0 → A else B."""
    digest = hashlib.sha256(f"{ro_seed}\0{question_id}".encode()).digest()
    return "A" if (digest[0] & 1) == 0 else "B"


def assign_order(
    *,
    order_regime: str,
    truthful_text: str,
    incorrect_text: str,
    question_id: str | None = None,
    ro_seed: int | None = None,
    belief_condition: str | None = None,
    belief_variant_id: str | None = None,
    trial_index: int | None = None,
) -> OrderAssignment:
    """Map truthful/incorrect texts onto A/B for the requested order regime.

    For RO, assignment depends only on ``(ro_seed, question_id)`` (DEC-009).
    ``belief_condition``, ``belief_variant_id``, and ``trial_index`` are accepted
    so callers can pass row context without changing the mapping.
    """
    del belief_condition, belief_variant_id, trial_index
    if order_regime == "CF":
        return OrderAssignment(
            candidate_a=truthful_text,
            candidate_b=incorrect_text,
            truthful_label="A",
            incorrect_label="B",
            order_regime="CF",
        )
    if order_regime == "IF":
        return OrderAssignment(
            candidate_a=incorrect_text,
            candidate_b=truthful_text,
            truthful_label="B",
            incorrect_label="A",
            order_regime="IF",
        )
    if order_regime == "RO":
        if question_id is None or ro_seed is None:
            raise ValueError("RO assignment requires question_id and ro_seed")
        truthful_label = _ro_truthful_label(ro_seed=ro_seed, question_id=question_id)
        if truthful_label == "A":
            return OrderAssignment(
                candidate_a=truthful_text,
                candidate_b=incorrect_text,
                truthful_label="A",
                incorrect_label="B",
                order_regime="RO",
                order_manifest_id=f"ro:primary:{ro_seed}",
            )
        return OrderAssignment(
            candidate_a=incorrect_text,
            candidate_b=truthful_text,
            truthful_label="B",
            incorrect_label="A",
            order_regime="RO",
            order_manifest_id=f"ro:primary:{ro_seed}",
        )
    raise ValueError(f"unsupported order_regime: {order_regime!r}")


def build_ro_manifest(*, ro_seed: int, question_ids: list[str]) -> dict[str, object]:
    """Build a primary RO assignment manifest for ``question_ids`` (DEC-009)."""
    assignments = {
        question_id: _ro_truthful_label(ro_seed=ro_seed, question_id=question_id)
        for question_id in question_ids
    }
    return {
        "ro_seed": ro_seed,
        "order_manifest_id": f"ro:primary:{ro_seed}",
        "ro_manifest_selection": "primary_single",
        "assignments": assignments,
    }


def hash_ro_manifest(manifest: dict[str, object]) -> str:
    """Stable SHA-256 hex digest of sorted RO assignments and identity fields."""
    assignments = manifest["assignments"]
    assert isinstance(assignments, dict)
    lines = [
        f"order_manifest_id={manifest['order_manifest_id']}",
        f"ro_seed={manifest['ro_seed']}",
        f"ro_manifest_selection={manifest['ro_manifest_selection']}",
    ]
    for question_id in sorted(assignments):
        lines.append(f"{question_id}={assignments[question_id]}")
    payload = "\n".join(lines).encode()
    return hashlib.sha256(payload).hexdigest()
