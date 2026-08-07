"""ORCH-038: ensure_selected_pool loads common_pool when coefficient_length < 1."""

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


def _study(*, artifact_dir: str, coefficient_length: int = 0) -> StudyConfig:
    feature_ids = ((17, 9),) if coefficient_length >= 1 else ()
    feature_scales = (2.0,) if coefficient_length >= 1 else ()
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
                layers=(17, 22),
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
            lambda_beta=0.01,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=feature_ids,
            feature_scales=feature_scales,
            coefficient_length=coefficient_length,
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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1",)),
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


def _write_pool(art: Path) -> None:
    (art / "feature_selection").mkdir(parents=True)
    (art / "feature_selection" / "common_pool.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "feature_ids": [[17, 1], [22, 3]],
                "feature_scales": [1.5, 0.5],
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
                    "22:3": {
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


@pytest.mark.unit
def test_adapters__ensure_selected_pool__empty_yaml__overlays_common_pool(
    tmp_path: Path,
) -> None:
    """ORCH-038: coefficient_length < 1 → load pool and populate feature fields."""
    from epistemic_sycophancy.runner.adapters.pool import ensure_selected_pool

    art = tmp_path / "art"
    _write_pool(art)
    study = _study(artifact_dir=str(art), coefficient_length=0)
    assert study.experiment.coefficient_length == 0
    assert study.experiment.feature_ids == ()

    overlaid = ensure_selected_pool(study)
    assert overlaid.experiment.feature_ids == ((17, 1), (22, 3))
    assert overlaid.experiment.feature_scales == (1.5, 0.5)
    assert overlaid.experiment.coefficient_length == 2
    # Original study unchanged.
    assert study.experiment.coefficient_length == 0


@pytest.mark.unit
def test_adapters__ensure_selected_pool__populated_yaml__returns_unchanged(
    tmp_path: Path,
) -> None:
    """ORCH-038: coefficient_length >= 1 → no pool load; same experiment features."""
    from epistemic_sycophancy.runner.adapters.pool import ensure_selected_pool

    art = tmp_path / "art"
    study = _study(artifact_dir=str(art), coefficient_length=1)
    overlaid = ensure_selected_pool(study)
    assert overlaid.experiment.feature_ids == ((17, 9),)
    assert overlaid.experiment.feature_scales == (2.0,)
    assert overlaid.experiment.coefficient_length == 1
