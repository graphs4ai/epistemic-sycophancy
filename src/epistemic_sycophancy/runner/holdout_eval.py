"""Holdout unlock stage (ORCH-015 / DEC-071)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.frozen import (
    FrozenExperimentConfig,
    freeze_experiment_config,
    mark_holdout_started,
)
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows


def _load_frozen(path: Path) -> FrozenExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hash_fields = {
        key: payload[key]
        for key in (
            "dataset_manifest_hash",
            "prompt_template_hash",
            "order_manifest_hash",
            "model_revision",
            "tokenizer_revision",
            "sae_revision",
            "hook_configuration_hash",
            "layer_set_hash",
            "study_yaml_fingerprint",
            "selected_features_hash",
            "feature_scales_hash",
            "objective_configuration_hash",
            "code_commit",
        )
    }
    return freeze_experiment_config(
        config_payload=payload.get("config_payload", {}),
        holdout_started=bool(payload.get("holdout_started", False)),
        **hash_fields,
    )


def run_holdout_eval_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    frozen_config_path: str,
    holdout_rows_provider: Callable[[], Sequence[Any]],
) -> dict[str, Any]:
    """Unlock holdout only after sealed freeze; mark_holdout_started (DEC-071)."""
    if freeze_status != "sealed":
        raise HoldoutAccessError(
            "holdout_eval requires freeze_status='sealed' "
            f"(got {freeze_status!r}; DEC-071)"
        )
    path = Path(frozen_config_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen config at {path}")
    frozen = _load_frozen(path)
    if frozen.freeze_status != "sealed":
        raise HoldoutAccessError("frozen config is not sealed")

    started = mark_holdout_started(frozen)
    rows = load_holdout_rows(
        freeze_status="sealed",
        frozen_config_artifact=started,
        rows_provider=holdout_rows_provider,
    )
    # Persist holdout_started on disk.
    payload = asdict(started)
    if "study_yaml_fingerprint" in json.loads(path.read_text(encoding="utf-8")):
        payload["study_yaml_fingerprint"] = json.loads(path.read_text(encoding="utf-8"))[
            "study_yaml_fingerprint"
        ]
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    out_dir = Path(study.run.artifact_dir) / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "holdout_rows.json"
    rows_path.write_text(
        json.dumps(list(rows), sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": {
            "holdout_started": True,
            "n_holdout_rows": len(list(rows)),
        },
        "artifacts": {"holdout": str(rows_path)},
    }
