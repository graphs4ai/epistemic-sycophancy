"""Optimizer checkpoint dump/load (Phase H OPT-009 / DEC-032)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CHECKPOINT_VERSION = "v1"
_ALLOWED_KINDS = frozenset({"cmaes", "projected_adam"})


def _to_plain(value: Any) -> Any:
    """Convert nested tensors / arrays to JSON-safe plain Python."""
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return value.detach().tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint value type: {type(value)!r}")


def dump_checkpoint(
    *,
    optimizer_kind: str,
    beta: Sequence[float],
    optimizer_state: Mapping[str, Any] | dict[str, Any],
    config_hash: str,
    objective_version: str,
    ro_manifest_hash: str,
) -> dict[str, Any]:
    """Serialize a versioned optimizer checkpoint (DEC-032)."""
    if optimizer_kind not in _ALLOWED_KINDS:
        raise ValueError(f"unsupported optimizer_kind: {optimizer_kind!r}")
    if not isinstance(config_hash, str) or len(config_hash) == 0:
        raise ValueError("config_hash must be a non-empty string")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "optimizer_kind": optimizer_kind,
        "beta": [float(v) for v in beta],
        "optimizer_state": _to_plain(optimizer_state),
        "config_hash": config_hash,
        "objective_version": objective_version,
        "ro_manifest_hash": ro_manifest_hash,
    }


def load_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Load a checkpoint and return a plain dict with DEC-032 fields."""
    required = (
        "checkpoint_version",
        "optimizer_kind",
        "beta",
        "optimizer_state",
        "config_hash",
        "objective_version",
        "ro_manifest_hash",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint_version: {checkpoint['checkpoint_version']!r}"
        )
    return dump_checkpoint(
        optimizer_kind=str(checkpoint["optimizer_kind"]),
        beta=list(checkpoint["beta"]),
        optimizer_state=dict(checkpoint["optimizer_state"]),
        config_hash=str(checkpoint["config_hash"]),
        objective_version=str(checkpoint["objective_version"]),
        ro_manifest_hash=str(checkpoint["ro_manifest_hash"]),
    )
