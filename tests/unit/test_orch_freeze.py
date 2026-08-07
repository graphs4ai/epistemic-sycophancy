"""ORCH-013: freeze stage seals FrozenExperimentConfig with study/stack hashes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudyFsCoverageConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study(artifact_dir: str) -> StudyConfig:
    return StudyConfig(
        stack=ExperimentStackConfig(
            model=ModelSpec(
                hf_id="google/gemma-3-4b-it",
                revision="093f9f388b31de276ce2de164bdc2081324b9767",
                tokenizer_revision="093f9f388b31de276ce2de164bdc2081324b9767",
                dtype="bfloat16",
                device_policy="cuda_required",
            ),
            sae=SaeSiteSpec(
                release="gemma-scope-2-4b-it-res",
                site="resid_post",
                width="width_65k",
                l0="l0_medium",
                layers=(17,),
            ),
            hooks=HookSpec(
                token_scope="last_prompt_token",
                resolver_id="gemma3_resid_post",
                k=None,
            ),
        ),
        experiment=ExperimentConfig(
            tau=1.0,
            lambda_n=0.0,
            lambda_c=0.0,
            lambda_beta=0.0,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=((17, 1),),
            feature_scales=(1.0,),
            coefficient_length=1,
            tie_policy="merge_into_q_minus",
            tie_band_epsilon=1e-6,
            mc1_tie_policy="fail_and_report",
            invalid_row_policy="fail_trial",
            multi_token_candidate_scoring="sum_log_probs",
            ro_manifest_selection="primary_single",
            continuation_A="A",
            continuation_B="B",
            continuation_include_eos=False,
            attribution_scope="last_prompt_token",
            pool_eligibility_override=False,
            pool_quota_per_list=8,
        ),
        run=StudyRunConfig(
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1")),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=0.1,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=2,
            ),
        ),
    )


def _study_empty_features(artifact_dir: str) -> StudyConfig:
    return StudyConfig(
        stack=ExperimentStackConfig(
            model=ModelSpec(
                hf_id="google/gemma-3-4b-it",
                revision="093f9f388b31de276ce2de164bdc2081324b9767",
                tokenizer_revision="093f9f388b31de276ce2de164bdc2081324b9767",
                dtype="bfloat16",
                device_policy="cuda_required",
            ),
            sae=SaeSiteSpec(
                release="gemma-scope-2-4b-it-res",
                site="resid_post",
                width="width_65k",
                l0="l0_medium",
                layers=(17,),
            ),
            hooks=HookSpec(
                token_scope="last_prompt_token",
                resolver_id="gemma3_resid_post",
                k=None,
            ),
        ),
        experiment=ExperimentConfig(
            tau=1.0,
            lambda_n=0.0,
            lambda_c=0.0,
            lambda_beta=0.0,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=(),
            feature_scales=(),
            coefficient_length=0,
            tie_policy="merge_into_q_minus",
            tie_band_epsilon=1e-6,
            mc1_tie_policy="fail_and_report",
            invalid_row_policy="fail_trial",
            multi_token_candidate_scoring="sum_log_probs",
            ro_manifest_selection="primary_single",
            continuation_A="A",
            continuation_B="B",
            continuation_include_eos=False,
            attribution_scope="last_prompt_token",
            pool_eligibility_override=False,
            pool_quota_per_list=8,
        ),
        run=StudyRunConfig(
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1")),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=0.1,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=2,
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__freeze__empty_yaml_features__overlays_pool_in_frozen_config(
    tmp_path: Path,
) -> None:
    """ORCH-013b: freeze seals populated feature_ids from common_pool (DEC-073/102)."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    art = tmp_path / "art"
    study = _study_empty_features(str(art))
    assert study.experiment.coefficient_length == 0
    (art / "feature_selection").mkdir(parents=True)
    (art / "feature_selection" / "common_pool.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "feature_ids": [[17, 1], [17, 7]],
                "feature_scales": [1.0, 2.0],
                "pool_size": 2,
                "scale_source": "decoder_norm",
                "provenance": {
                    "17:1": {
                        "nominators": [
                            {
                                "order": "CF",
                                "component": "resistance",
                                "signed_jacobian": 1.0,
                            }
                        ],
                        "surrogates": {},
                    },
                    "17:7": {
                        "nominators": [
                            {
                                "order": "CF",
                                "component": "recovery",
                                "signed_jacobian": 2.0,
                            }
                        ],
                        "surrogates": {},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = dispatch_stage("freeze", study=study, freeze_status="unsealed")
    assert result.ok is True
    path = Path(result.artifacts["frozen_config"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config_payload"]
    assert config["feature_ids"] == [[17, 1], [17, 7]]
    assert config["coefficient_length"] == 2
    # YAML study remains empty (in-memory overlay only).
    assert study.experiment.coefficient_length == 0


@pytest.mark.unit
def test_dispatch__freeze__seals_frozen_experiment_config_with_study_and_stack_hashes(
    tmp_path: Path,
) -> None:
    """ORCH-013: freeze writes sealed FrozenExperimentConfig (DEC-044/070)."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    study = _study(str(tmp_path / "art"))
    result = dispatch_stage("freeze", study=study, freeze_status="unsealed")
    assert result.ok is True
    assert result.metrics.get("freeze_status") == "sealed"
    assert "frozen_config" in result.artifacts
    path = Path(result.artifacts["frozen_config"])
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["freeze_status"] == "sealed"
    assert payload["holdout_started"] is False
    assert payload["study_yaml_fingerprint"]
    assert payload["model_revision"] == study.stack.model.revision
    assert payload["layer_set_hash"]
    assert payload["hook_configuration_hash"]
