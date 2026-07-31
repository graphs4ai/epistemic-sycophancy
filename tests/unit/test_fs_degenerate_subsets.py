"""FSC-008: empty component subset skip vs dual-empty raise (DEC-085)."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.metrics.exceptions import DegenerateBaselineError
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.runner.fs_dispatch import run_feature_selection_dispatch
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
            pool_quota_per_list=2,
        ),
        run=StudyRunConfig(
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=8,
            prompt_batch_size=1,
            smoke=StudySmokeConfig(question_ids=("q1", "q2")),
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
                max_steps=1,
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_fs_dispatch__empty_recovery_subset__skips_with_recorded_count(
    tmp_path: Path,
) -> None:
    """FSC-008: empty recovery list is skipped; resistance still builds pool."""

    def jacobian_fn(*, order_regime, question_ids, component: str):
        del order_regime, question_ids
        if component == "resistance":
            return {(17, 1): 2.0, (17, 2): 1.0}
        if component == "recovery":
            return {}  # empty Q- / skipped
        return {(17, 1): 0.5}

    result = run_feature_selection_dispatch(
        study=_study(str(tmp_path / "art")),
        freeze_status="unsealed",
        jacobian_fn=jacobian_fn,
        scale_fn=lambda keys: {k: 1.0 for k in keys},
        question_ids=("q1", "q2"),
        optimization_question_ids=("q_opt",),
    )
    assert result["component_jacobians"][("CF", "recovery")] == {}
    skipped = result["metrics"]["component_skips"]
    assert skipped[("CF", "recovery")]["skipped"] is True
    assert skipped[("CF", "recovery")]["n_prompts"] == 0
    assert skipped[("CF", "resistance")]["skipped"] is False
    assert set(result["pool"].feature_ids) == {(17, 1), (17, 2)}


@pytest.mark.unit
def test_fs_dispatch__both_resistance_and_recovery_empty__raises(
    tmp_path: Path,
) -> None:
    """FSC-008 / DEC-085: both behavior lists empty → DegenerateBaselineError."""

    def jacobian_fn(*, order_regime, question_ids, component: str):
        del order_regime, question_ids, component
        return {}

    with pytest.raises(DegenerateBaselineError, match="resistance|recovery"):
        run_feature_selection_dispatch(
            study=_study(str(tmp_path / "art")),
            freeze_status="unsealed",
            jacobian_fn=jacobian_fn,
            scale_fn=lambda keys: {k: 1.0 for k in keys},
            question_ids=("q1", "q2"),
            optimization_question_ids=("q_opt",),
        )
