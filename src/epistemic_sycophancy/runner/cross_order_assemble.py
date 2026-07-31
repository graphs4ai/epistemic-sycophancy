"""Cross-study 3×3 assemble stage (ORDER-EXP / DEC-087)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from epistemic_sycophancy.evaluation.cross_order import build_cross_order_matrix
from epistemic_sycophancy.logging.pipeline import log_progress
from epistemic_sycophancy.optimization.checkpoint import load_checkpoint

_ORDERS = ("CF", "IF", "RO")


@dataclass(frozen=True)
class CrossOrderCampaignConfig:
    """Paths to three sealed single-order studies plus assemble output dir."""

    sources: Mapping[str, str]
    artifact_dir: str

    def __post_init__(self) -> None:
        missing = [o for o in _ORDERS if o not in self.sources]
        if missing:
            raise ValueError(
                "cross_order campaign sources must include CF, IF, and RO; "
                f"missing {missing}"
            )
        extra = sorted(set(self.sources) - set(_ORDERS))
        if extra:
            raise ValueError(f"unexpected cross_order source keys: {extra}")
        if not str(self.artifact_dir).strip():
            raise ValueError("cross_order campaign artifact_dir must be nonempty")


def _load_source_beta(source_root: Path, *, expected_order: str) -> tuple[float, ...]:
    freeze_path = source_root / "freeze" / "frozen_experiment_config.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(
            f"cross_order source {expected_order!r} missing sealed freeze at {freeze_path}"
        )
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    # Accept either sealed marker or presence of frozen payload.
    status = frozen.get("freeze_status", "sealed")
    if status != "sealed":
        raise ValueError(
            f"cross_order source {expected_order!r} freeze_status={status!r} "
            "(require sealed)"
        )
    order = str(frozen.get("order_regime") or frozen.get("run", {}).get("order_regime", ""))
    if order and order != expected_order:
        raise ValueError(
            f"cross_order source key {expected_order!r} has order_regime={order!r}"
        )
    ckpt_path = source_root / "optimize" / "best_checkpoint.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"cross_order source {expected_order!r} missing checkpoint at {ckpt_path}"
        )
    ckpt = load_checkpoint(json.loads(ckpt_path.read_text(encoding="utf-8")))
    return tuple(float(x) for x in ckpt["beta"])


def _load_partition_fingerprint(source_root: Path, *, order: str) -> str:
    path = source_root / "baseline" / f"partition_{order}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"cross_order source {order!r} missing baseline partition at {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("fingerprint", f"missing-fingerprint-{order}"))


def run_cross_order_assemble(
    *,
    campaign: CrossOrderCampaignConfig,
    metrics_by_evaluated_under: Mapping[str, Mapping[str, float | int]],
    optimization_order_manifest_hashes: Mapping[str, str] | None = None,
    evaluation_order_manifest_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble DEC-041 3×3 matrix from three sealed single-order studies."""
    betas: dict[str, Sequence[float]] = {}
    fingerprints: dict[str, str] = {}
    sources_payload: dict[str, Any] = {}
    for order in _ORDERS:
        root = Path(campaign.sources[order])
        beta = _load_source_beta(root, expected_order=order)
        betas[order] = beta
        fingerprints[order] = _load_partition_fingerprint(root, order=order)
        sources_payload[order] = {
            "artifact_dir": str(root),
            "beta": list(beta),
            "baseline_partition_fingerprint": fingerprints[order],
        }

    opt_hashes = dict(optimization_order_manifest_hashes or {})
    eval_hashes = dict(evaluation_order_manifest_hashes or {})
    for order in _ORDERS:
        opt_hashes.setdefault(order, f"opt-{order}")
        eval_hashes.setdefault(order, f"eval-{order}")

    cells = build_cross_order_matrix(
        betas_by_optimized_under=betas,
        optimization_order_manifest_hashes=opt_hashes,
        evaluation_order_manifest_hashes=eval_hashes,
        baseline_partition_fingerprints=fingerprints,
        metrics_by_evaluated_under=metrics_by_evaluated_under,
    )

    out_dir = Path(campaign.artifact_dir) / "cross_order"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_path = out_dir / "sources.json"
    matrix_path = out_dir / "cross_order_matrix.json"
    sources_path.write_text(
        json.dumps({"sources": sources_payload}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps({"cells": [asdict(c) for c in cells]}, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    log_progress(
        "cross_order_assemble",
        n_cells=len(cells),
        sources=",".join(_ORDERS),
        path=str(matrix_path),
    )
    return {
        "metrics": {"n_cells": len(cells)},
        "artifacts": {
            "sources": str(sources_path),
            "cross_order_matrix": str(matrix_path),
        },
        "cells": cells,
    }
