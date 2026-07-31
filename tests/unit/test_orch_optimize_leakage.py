"""ORCH-010: optimize never reads holdout/validation for parameter updates."""

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
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
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
            smoke=StudySmokeConfig(question_ids=("q1",)),
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
                max_steps=2,
                question_ids=("qo1", "qo2"),
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__optimize__never_reads_holdout_or_validation_for_parameter_updates(
    tmp_path: Path,
) -> None:
    """ORCH-010: objective/grad only see optimization IDs; holdout trap unused."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    accessed: list[str] = []

    def objective_fn(beta, question_ids):
        for qid in question_ids:
            if str(qid).startswith("qh") or str(qid).startswith("qv"):
                raise HoldoutAccessError(f"optimize must not use {qid}")
            accessed.append(str(qid))
        return float(sum(beta))

    def grad_fn(beta, question_ids):
        for qid in question_ids:
            if str(qid).startswith("qh") or str(qid).startswith("qv"):
                raise HoldoutAccessError(f"optimize must not use {qid}")
        return tuple(0.0 for _ in beta)

    result = dispatch_stage(
        "optimize",
        study=_study(str(tmp_path / "art")),
        freeze_status="unsealed",
        identity_passed=True,
        objective_fn=objective_fn,
        grad_fn=grad_fn,
        optimization_question_ids=("qo1", "qo2"),
        validation_question_ids=("qv1",),
        holdout_question_ids=("qh1",),
    )
    assert result.ok is True
    assert accessed
    assert all(q.startswith("qo") for q in accessed)
    assert "qv1" not in accessed
    assert "qh1" not in accessed
