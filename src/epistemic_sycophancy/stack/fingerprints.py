"""Stack pin fingerprints for result artifacts (Phase K RUN-014)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any


def _sha256_hex(payload: str) -> str:
    return sha256(payload.encode("utf-8")).hexdigest()


def build_stack_fingerprint_fields(
    *,
    model_revision: str,
    tokenizer_revision: str,
    sae_revision: str,
    hook_configuration: Mapping[str, Any],
    layers: Sequence[int],
    dataset_manifest_hash: str,
    prompt_template_hash: str,
    order_manifest_hash: str,
    selected_features_hash: str,
    feature_scales_hash: str,
    objective_configuration_hash: str,
    code_commit: str,
    study_yaml_fingerprint: str,
) -> dict[str, str]:
    """Build REPRO-001 fields including layer_set_hash and study YAML fingerprint."""
    layer_set_hash = _sha256_hex(",".join(str(layer) for layer in layers))
    hook_configuration_hash = _sha256_hex(
        json.dumps(dict(hook_configuration), sort_keys=True, separators=(",", ":"))
    )
    return {
        "dataset_manifest_hash": dataset_manifest_hash,
        "prompt_template_hash": prompt_template_hash,
        "order_manifest_hash": order_manifest_hash,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "sae_revision": sae_revision,
        "hook_configuration_hash": hook_configuration_hash,
        "layer_set_hash": layer_set_hash,
        "study_yaml_fingerprint": study_yaml_fingerprint,
        "selected_features_hash": selected_features_hash,
        "feature_scales_hash": feature_scales_hash,
        "objective_configuration_hash": objective_configuration_hash,
        "code_commit": code_commit,
    }
