"""Frozen experiment configuration (Phase J REPRO-003 / DEC-044)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Mapping

from epistemic_sycophancy.reproducibility.artifacts import REQUIRED_RESULT_HASH_FIELDS


class ConfigImmutabilityError(Exception):
    """Raised when a holdout-started frozen config is mutated."""


@dataclass(frozen=True)
class FrozenExperimentConfig:
    """Sealed experiment configuration artifact (DEC-044)."""

    freeze_status: str
    config_fingerprint: str
    holdout_started: bool
    config_payload: Mapping[str, Any]
    dataset_manifest_hash: str
    prompt_template_hash: str
    order_manifest_hash: str
    model_revision: str
    tokenizer_revision: str
    sae_revision: str
    hook_configuration_hash: str
    selected_features_hash: str
    feature_scales_hash: str
    objective_configuration_hash: str
    code_commit: str

    def replace_field(self, field_name: str, value: Any) -> FrozenExperimentConfig:
        """Attempt to update a payload field; rejected after holdout start."""
        if self.holdout_started:
            raise ConfigImmutabilityError(
                "FrozenExperimentConfig mutation rejected after holdout start "
                "(REPRO-003 / DEC-044)"
            )
        if field_name not in self.config_payload:
            raise KeyError(field_name)
        new_payload = dict(self.config_payload)
        new_payload[field_name] = value
        return freeze_experiment_config(
            config_payload=new_payload,
            dataset_manifest_hash=self.dataset_manifest_hash,
            prompt_template_hash=self.prompt_template_hash,
            order_manifest_hash=self.order_manifest_hash,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            sae_revision=self.sae_revision,
            hook_configuration_hash=self.hook_configuration_hash,
            selected_features_hash=self.selected_features_hash,
            feature_scales_hash=self.feature_scales_hash,
            objective_configuration_hash=self.objective_configuration_hash,
            code_commit=self.code_commit,
            holdout_started=False,
        )


def freeze_experiment_config(
    *,
    config_payload: Mapping[str, Any],
    holdout_started: bool = False,
    **hash_fields: str,
) -> FrozenExperimentConfig:
    """Build a sealed FrozenExperimentConfig with REPRO-001 hash fields."""
    missing = [name for name in REQUIRED_RESULT_HASH_FIELDS if name not in hash_fields]
    if missing:
        raise ValueError(f"missing required freeze hash fields: {missing}")
    for name in REQUIRED_RESULT_HASH_FIELDS:
        value = hash_fields[name]
        if value is None or str(value) == "":
            raise ValueError(f"required freeze hash field {name!r} must be non-empty")
    material = json.dumps(dict(config_payload), sort_keys=True, default=str)
    fingerprint = sha256(material.encode("utf-8")).hexdigest()
    return FrozenExperimentConfig(
        freeze_status="sealed",
        config_fingerprint=fingerprint,
        holdout_started=bool(holdout_started),
        config_payload=dict(config_payload),
        dataset_manifest_hash=str(hash_fields["dataset_manifest_hash"]),
        prompt_template_hash=str(hash_fields["prompt_template_hash"]),
        order_manifest_hash=str(hash_fields["order_manifest_hash"]),
        model_revision=str(hash_fields["model_revision"]),
        tokenizer_revision=str(hash_fields["tokenizer_revision"]),
        sae_revision=str(hash_fields["sae_revision"]),
        hook_configuration_hash=str(hash_fields["hook_configuration_hash"]),
        selected_features_hash=str(hash_fields["selected_features_hash"]),
        feature_scales_hash=str(hash_fields["feature_scales_hash"]),
        objective_configuration_hash=str(hash_fields["objective_configuration_hash"]),
        code_commit=str(hash_fields["code_commit"]),
    )


def mark_holdout_started(frozen: FrozenExperimentConfig) -> FrozenExperimentConfig:
    """Return a sealed copy with holdout_started=True."""
    if frozen.freeze_status != "sealed":
        raise ConfigImmutabilityError(
            f"holdout start requires sealed freeze_status; got {frozen.freeze_status!r}"
        )
    return replace(frozen, holdout_started=True)
