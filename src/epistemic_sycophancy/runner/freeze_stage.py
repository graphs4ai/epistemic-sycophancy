"""Freeze stage: StudyConfig → sealed FrozenExperimentConfig (ORCH-013)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.frozen import freeze_experiment_config
from epistemic_sycophancy.config.load_study import (
    study_config_fingerprint,
)
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.logging.pipeline import log_audit
from epistemic_sycophancy.runner.cli import _stage_hash_fields


def run_freeze_dispatch(*, study: StudyConfig) -> dict[str, Any]:
    """Seal study into FrozenExperimentConfig artifact under artifact_dir/freeze/."""
    hashes = _stage_hash_fields(study)
    fingerprint = study_config_fingerprint(study)
    config_payload = {
        "study_yaml_fingerprint": fingerprint,
        "model_revision": study.stack.model.revision,
        "tokenizer_revision": study.stack.model.tokenizer_revision,
        "layers": list(study.stack.sae.layers),
        "feature_ids": [
            list(fid) if isinstance(fid, tuple) else fid
            for fid in study.experiment.feature_ids
        ],
        "coefficient_length": study.experiment.coefficient_length,
        "optimize": {
            "budget_match_on": study.run.optimize.budget_match_on,
            "max_steps": study.run.optimize.max_steps,
            "n_trials": study.run.optimize.n_trials,
            "population_size": study.run.optimize.population_size,
        },
    }
    frozen = freeze_experiment_config(
        config_payload=config_payload,
        holdout_started=False,
        dataset_manifest_hash="orch-dataset",
        prompt_template_hash="orch-prompt",
        order_manifest_hash="orch-order",
        model_revision=hashes["model_revision"],
        tokenizer_revision=study.stack.model.tokenizer_revision,
        sae_revision=hashes["sae_revision"],
        hook_configuration_hash=hashes["hook_configuration_hash"],
        layer_set_hash=hashes["layer_set_hash"],
        study_yaml_fingerprint=hashes["study_yaml_fingerprint"],
        selected_features_hash="orch-selected-features",
        feature_scales_hash="orch-feature-scales",
        objective_configuration_hash="orch-objective",
        code_commit="orch-local",
    )
    out_dir = Path(study.run.artifact_dir) / "freeze"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frozen_experiment_config.json"
    payload = asdict(frozen)
    payload["study_yaml_fingerprint"] = hashes["study_yaml_fingerprint"]
    payload["order_regime"] = study.run.order_regime
    payload["freeze_status"] = frozen.freeze_status
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    log_audit(
        "freeze_sealed",
        freeze_status=frozen.freeze_status,
        holdout_started=frozen.holdout_started,
        config_fingerprint=frozen.config_fingerprint,
        path=str(path),
    )
    return {
        "frozen": frozen,
        "metrics": {
            "freeze_status": frozen.freeze_status,
            "holdout_started": frozen.holdout_started,
            "config_fingerprint": frozen.config_fingerprint,
        },
        "artifacts": {"frozen_config": str(path)},
    }
