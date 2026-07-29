"""Result artifact hash completeness (Phase I REPRO-001)."""

from __future__ import annotations

from typing import Any

REQUIRED_RESULT_HASH_FIELDS: tuple[str, ...] = (
    "dataset_manifest_hash",
    "prompt_template_hash",
    "order_manifest_hash",
    "model_revision",
    "tokenizer_revision",
    "sae_revision",
    "hook_configuration_hash",
    "selected_features_hash",
    "feature_scales_hash",
    "objective_configuration_hash",
    "code_commit",
)


def build_result_artifact_hashes(**fields: Any) -> dict[str, str]:
    """Build a result-artifact hash map; require the REPRO-001 field set."""
    missing = [name for name in REQUIRED_RESULT_HASH_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"missing required result artifact hash fields: {missing}")
    result: dict[str, str] = {}
    for name in REQUIRED_RESULT_HASH_FIELDS:
        value = fields[name]
        if value is None or str(value) == "":
            raise ValueError(f"required result artifact field {name!r} must be non-empty")
        result[name] = str(value)
    return result
