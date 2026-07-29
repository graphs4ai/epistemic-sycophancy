"""RUN-014: stack artifact fingerprints include model/SAE/hook/layer-set hashes."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.reproducibility.artifacts import (
    REQUIRED_RESULT_HASH_FIELDS,
    build_result_artifact_hashes,
)
from epistemic_sycophancy.stack.fingerprints import build_stack_fingerprint_fields


@pytest.mark.unit
def test_runner__artifacts__include_model_sae_hook_layer_set_hashes() -> None:
    """RUN-014: REPRO-001 set plus layer_set_hash from stack pin."""
    assert "layer_set_hash" in REQUIRED_RESULT_HASH_FIELDS
    fields = build_stack_fingerprint_fields(
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        tokenizer_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        sae_revision="3e94b68be95290aada5b7525cf431d3040f81bb1",
        hook_configuration={"resolver_id": "gemma3_resid_post", "token_scope": "last_prompt_token"},
        layers=(9, 17, 22, 29),
        dataset_manifest_hash="d" * 64,
        prompt_template_hash="p" * 64,
        order_manifest_hash="o" * 64,
        selected_features_hash="f" * 64,
        feature_scales_hash="s" * 64,
        objective_configuration_hash="j" * 64,
        code_commit="c" * 40,
        study_yaml_fingerprint="y" * 64,
    )
    artifact = build_result_artifact_hashes(**fields)
    assert artifact["model_revision"]
    assert artifact["sae_revision"]
    assert artifact["hook_configuration_hash"]
    assert artifact["layer_set_hash"]
    assert len(artifact["layer_set_hash"]) == 64


@pytest.mark.unit
def test_runner__artifacts__include_study_yaml_and_stack_hashes() -> None:
    """WIRE-012: artifacts include study_yaml_fingerprint + stack hashes."""
    assert "study_yaml_fingerprint" in REQUIRED_RESULT_HASH_FIELDS
    fields = build_stack_fingerprint_fields(
        model_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        tokenizer_revision="093f9f388b31de276ce2de164bdc2081324b9767",
        sae_revision="3e94b68be95290aada5b7525cf431d3040f81bb1",
        hook_configuration={"resolver_id": "gemma3_resid_post", "token_scope": "last_prompt_token"},
        layers=(17,),
        dataset_manifest_hash="d" * 64,
        prompt_template_hash="p" * 64,
        order_manifest_hash="o" * 64,
        selected_features_hash="f" * 64,
        feature_scales_hash="s" * 64,
        objective_configuration_hash="j" * 64,
        code_commit="c" * 40,
        study_yaml_fingerprint="a" * 64,
    )
    artifact = build_result_artifact_hashes(**fields)
    assert artifact["study_yaml_fingerprint"] == "a" * 64
    assert artifact["layer_set_hash"]
