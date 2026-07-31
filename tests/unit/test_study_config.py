"""StudyConfig schema validation (Phase L CFGFILE-001 / DEC-056)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig, InvalidExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _valid_stack() -> ExperimentStackConfig:
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


def _valid_experiment_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "tau": 1.0,
        "lambda_n": 0.0,
        "lambda_c": 0.0,
        "lambda_beta": 0.0,
        "delta_n": 0.0,
        "delta_c": 0.0,
        "w_r": 0.5,
        "w_u": 0.5,
        "beta_lower": -2.0,
        "beta_upper": 0.0,
        "feature_ids": (),
        "feature_scales": (),
        "coefficient_length": 0,
        "tie_policy": "merge_into_q_minus",
        "tie_band_epsilon": 1e-6,
        "mc1_tie_policy": "fail_and_report",
        "invalid_row_policy": "fail_trial",
        "multi_token_candidate_scoring": "sum_log_probs",
        "ro_manifest_selection": "primary_single",
        "continuation_A": "A",
        "continuation_B": "B",
        "continuation_include_eos": False,
        "attribution_scope": "last_prompt_token",
        "pool_eligibility_override": False,
        "pool_quota_per_list": 8,
    }
    kwargs.update(overrides)
    return kwargs


def _valid_run() -> StudyRunConfig:
    return StudyRunConfig(
        artifact_dir="artifacts/first_study",
        order_regime="CF",
        feature_chunk_size=1024,
        prompt_batch_size=1,
        smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
        optimizer=StudyOptimizerConfig(
            kind="projected_adam",
            adam_lr=0.1,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_eps=1e-8,
            adam_microbatch_questions=1,
            max_steps=1,
        ),
        optimize=StudyOptimizeConfig(
            budget_match_on="n_objective_evals",
            max_steps=20,
            n_questions=4,
        ),
    )


@pytest.mark.unit
def test_study_run_config__order_regime__must_be_single_cf_if_or_ro() -> None:
    """ORDER-EXP-001: each study has exactly one order_regime in {CF, IF, RO}."""
    from epistemic_sycophancy.config.study import study_order_regime

    for regime in ("CF", "IF", "RO"):
        run = StudyRunConfig(
            artifact_dir="artifacts/first_study",
            order_regime=regime,
            feature_chunk_size=1024,
            prompt_batch_size=1,
            smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=0.1,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
                max_steps=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=20,
                n_questions=4,
            ),
        )
        assert run.order_regime == regime
        study = StudyConfig(
            stack=_valid_stack(),
            experiment=ExperimentConfig(**_valid_experiment_kwargs()),
            run=run,
        )
        assert study_order_regime(study) == regime

    with pytest.raises(InvalidExperimentConfig, match="order_regime"):
        StudyRunConfig(
            artifact_dir="artifacts/first_study",
            order_regime="XX",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=0.1,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
                max_steps=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=20,
                n_questions=4,
            ),
        )


@pytest.mark.unit
def test_study_config__missing_required_policy_field__raises_invalid_config() -> None:
    """CFGFILE-001: StudyConfig rejects missing CFG-006 policy fields (DEC-056)."""
    with pytest.raises(InvalidExperimentConfig, match="tie_policy"):
        StudyConfig(
            stack=_valid_stack(),
            experiment=ExperimentConfig(**_valid_experiment_kwargs(tie_policy=None)),
            run=_valid_run(),
        )

    with pytest.raises(InvalidExperimentConfig, match="artifact_dir|run"):
        StudyConfig(
            stack=_valid_stack(),
            experiment=ExperimentConfig(**_valid_experiment_kwargs()),
            run=StudyRunConfig(
                artifact_dir=None,  # type: ignore[arg-type]
                order_regime="CF",
                feature_chunk_size=1024,
                prompt_batch_size=1,
                smoke=StudySmokeConfig(
                    n_questions=2, split="feature_selection", seed=0
                ),
                optimizer=StudyOptimizerConfig(
                    kind="projected_adam",
                    adam_lr=0.1,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    adam_eps=1e-8,
                    adam_microbatch_questions=1,
                    max_steps=1,
                ),
                optimize=StudyOptimizeConfig(
                    budget_match_on="n_objective_evals",
                    max_steps=20,
                    n_questions=4,
                ),
            ),
        )

    study = StudyConfig(
        stack=_valid_stack(),
        experiment=ExperimentConfig(**_valid_experiment_kwargs()),
        run=_valid_run(),
    )
    assert study.stack.sae.layers == (17,)
    assert study.experiment.tie_policy == "merge_into_q_minus"
    assert study.run.smoke.n_questions == 2
