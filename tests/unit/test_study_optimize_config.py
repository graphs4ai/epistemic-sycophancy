"""ORCH-008: StudyConfig.run.optimize non-smoke budgets (DEC-066/068)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig, InvalidExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _stack() -> ExperimentStackConfig:
    return ExperimentStackConfig(
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
    )


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(
        tau=1.0,
        lambda_n=1.0,
        lambda_c=1.0,
        lambda_beta=0.01,
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
    )


def _optimizer() -> StudyOptimizerConfig:
    return StudyOptimizerConfig(
        kind="projected_adam",
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        max_steps=1,
    )


@pytest.mark.unit
def test_study_config__run_optimize__requires_explicit_non_smoke_budgets_distinct_from_smoke() -> None:
    """ORCH-008: run.optimize required; budgets distinct from smoke max_steps (DEC-066)."""
    with pytest.raises(TypeError):
        # Missing optimize kwarg must fail construction (CFG-006).
        StudyRunConfig(  # type: ignore[call-arg]
            artifact_dir="artifacts/x",
            order_regimes=("CF",),
            feature_chunk_size=1024,
            prompt_batch_size=1,
            smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
            optimizer=_optimizer(),
        )

    with pytest.raises(InvalidExperimentConfig, match="optimize|budget_match_on|max_steps"):
        StudyOptimizeConfig(budget_match_on="n_objective_evals")

    optimize = StudyOptimizeConfig(
        budget_match_on="n_objective_evals",
        max_steps=20,
        n_questions=4,
    )
    assert optimize.max_steps == 20
    assert optimize.n_questions == 4
    assert optimize.question_ids is None
    assert optimize.max_steps != _optimizer().max_steps

    run = StudyRunConfig(
        artifact_dir="artifacts/x",
        order_regimes=("CF",),
        feature_chunk_size=1024,
        prompt_batch_size=1,
        smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
        optimizer=_optimizer(),
        optimize=optimize,
    )
    study = StudyConfig(stack=_stack(), experiment=_experiment(), run=run)
    assert study.run.optimize.max_steps == 20
    assert study.run.optimizer.max_steps == 1

    # XOR coverage: both question_ids and n_questions forbidden.
    with pytest.raises(InvalidExperimentConfig, match="question_ids|n_questions"):
        StudyOptimizeConfig(
            budget_match_on="n_objective_evals",
            max_steps=5,
            n_questions=2,
            question_ids=("q1",),
        )

    # CMA fields when used without Adam max_steps path.
    cma_opt = StudyOptimizeConfig(
        budget_match_on="n_forward_equiv",
        n_trials=10,
        population_size=8,
    )
    assert cma_opt.n_trials == 10
    assert cma_opt.population_size == 8
