"""load_study_config fingerprint round-trip (Phase L CFGFILE-002 / DEC-057)."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.config.load_study import (
    load_study_config,
    study_config_fingerprint,
)


_MINIMAL_STUDY_YAML = """\
stack:
  model:
    hf_id: google/gemma-3-4b-it
    revision: 093f9f388b31de276ce2de164bdc2081324b9767
    tokenizer_revision: 093f9f388b31de276ce2de164bdc2081324b9767
    dtype: bfloat16
    device_policy: cuda_required
  sae:
    release: gemma-scope-2-4b-it-res
    site: resid_post
    width: width_65k
    l0: l0_medium
    layers: [17]
  hooks:
    token_scope: last_prompt_token
    resolver_id: gemma3_resid_post
    k: null
experiment:
  tau: 1.0
  lambda_n: 0.0
  lambda_c: 0.0
  lambda_beta: 0.0
  delta_n: 0.0
  delta_c: 0.0
  w_r: 0.5
  w_u: 0.5
  beta_lower: -2.0
  beta_upper: 0.0
  feature_ids: []
  feature_scales: []
  coefficient_length: 0
  tie_policy: merge_into_q_minus
  tie_band_epsilon: 1.0e-6
  mc1_tie_policy: fail_and_report
  invalid_row_policy: fail_trial
  multi_token_candidate_scoring: sum_log_probs
  ro_manifest_selection: primary_single
  continuation_A: A
  continuation_B: B
  continuation_include_eos: false
  attribution_scope: last_prompt_token
  pool_eligibility_override: false
  pool_quota_per_list: 8
run:
  artifact_dir: artifacts/first_study
  order_regimes: [CF, IF, RO]
  feature_chunk_size: 1024
  prompt_batch_size: 1
  smoke:
    n_questions: 2
    split: feature_selection
    seed: 0
  optimizer:
    kind: projected_adam
    adam_lr: 0.1
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-8
    adam_microbatch_questions: 1
    max_steps: 1
"""


@pytest.mark.unit
def test_load_study_config__identical_files__stable_fingerprint_round_trip(
    tmp_path: Path,
) -> None:
    """CFGFILE-002: identical YAML files yield the same study fingerprint."""
    path_a = tmp_path / "study_a.yaml"
    path_b = tmp_path / "study_b.yaml"
    path_a.write_text(_MINIMAL_STUDY_YAML, encoding="utf-8")
    path_b.write_text(_MINIMAL_STUDY_YAML, encoding="utf-8")

    study_a = load_study_config(path_a)
    study_b = load_study_config(path_b)
    fp_a = study_config_fingerprint(study_a)
    fp_b = study_config_fingerprint(study_b)

    assert fp_a == fp_b
    assert len(fp_a) == 64
    assert study_a.stack.sae.layers == (17,)
    assert study_a.experiment.pool_quota_per_list == 8
    assert study_a.run.optimizer.kind == "projected_adam"
