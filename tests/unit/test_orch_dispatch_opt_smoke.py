"""ORCH-005: dispatch opt_smoke finite objective + optional Adam step."""

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
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study(*, artifact_dir: str) -> StudyConfig:
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
            order_regimes=("CF",),
            feature_chunk_size=1024,
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
                max_steps=20,
                n_questions=4,
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__opt_smoke__finite_objective_optional_adam_step_identity_gated(
    tmp_path: Path,
) -> None:
    """ORCH-005: opt_smoke evaluates finite L; optional Adam; identity gated."""
    from epistemic_sycophancy.reproducibility.phase_gates import OptimizationBlockedError
    from epistemic_sycophancy.runner.cli import dispatch_stage

    study = _study(artifact_dir=str(tmp_path / "art"))
    margin_payload = {
        "ib_margins_by_question": {"q1": [1.0], "q2": [0.5]},
        "cb_margins_by_question": {"q1": [0.8], "q2": [-0.2]},
        "baseline_cb_margins": {"q1": [0.8], "q2": [-0.2]},
        "baseline_neutral_margins": {"q1": 1.0, "q2": -0.5},
        "current_neutral_margins": {"q1": 1.0, "q2": -0.5},
        "q_plus": ("q1",),
        "q_minus": ("q2",),
    }

    result = dispatch_stage(
        "opt_smoke",
        study=study,
        freeze_status="unsealed",
        identity_passed=True,
        margin_payload=margin_payload,
        beta=(0.0,),
        adam_grad=(0.5,),
    )
    assert result.ok is True
    assert result.metrics.get("l_total") is not None
    assert float(result.metrics["l_total"]) == float(result.metrics["l_total"])  # finite
    assert "opt_smoke" in result.artifacts
    assert result.metrics.get("beta_after") is not None
    beta_after = result.metrics["beta_after"]
    assert all(study.experiment.beta_lower <= b <= study.experiment.beta_upper for b in beta_after)

    with pytest.raises(OptimizationBlockedError):
        dispatch_stage(
            "opt_smoke",
            study=study,
            freeze_status="unsealed",
            identity_passed=False,
            margin_payload=margin_payload,
            beta=(0.0,),
        )
