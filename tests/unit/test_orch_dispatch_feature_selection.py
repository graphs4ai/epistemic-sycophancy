"""ORCH-004: dispatch feature_selection with Jacobians, pool, decoder_norm scales."""

from __future__ import annotations

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
        ),
        run=StudyRunConfig(
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1", "q2")),
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
                max_steps=20,
                n_questions=4,
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__feature_selection__stack_jacobians_pool_and_decoder_norm_no_leakage(
    tmp_path: Path,
) -> None:
    """ORCH-004: FS Jacobians → pool + scales; no opt/val/holdout leakage."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    study = _study(artifact_dir=str(tmp_path / "art"))
    seen: list[tuple[str, tuple[str, ...]]] = []

    def jacobian_fn(*, order_regime: str, question_ids: tuple[str, ...], component: str = "resistance"):
        seen.append((order_regime, question_ids, component))
        return {(17, 3): 1.5, (17, 1): 0.25, (17, 9): -0.1}

    def scale_fn(keys):
        return {key: 1.0 + 0.1 * key[1] for key in keys}

    result = dispatch_stage(
        "feature_selection",
        study=study,
        freeze_status="unsealed",
        jacobian_fn=jacobian_fn,
        scale_fn=scale_fn,
        optimization_question_ids=("qo1",),
        validation_question_ids=("qv1",),
        holdout_question_ids=("qh1",),
    )

    assert result.ok is True
    assert seen and all(qids == ("q1", "q2") for _, qids, _comp in seen)
    assert result.metrics.get("pool_size", 0) >= 1
    assert "pool" in result.artifacts
    pool_path = Path(result.artifacts["pool"])
    assert pool_path.is_file()
    text = pool_path.read_text(encoding="utf-8")
    assert "qo1" not in text
    assert "qv1" not in text
    assert "qh1" not in text
    assert "17" in text  # layer present
    assert result.metrics.get("scale_source") == "decoder_norm"
    payload = __import__("json").loads(text)
    assert payload.get("schema_version") == 2
