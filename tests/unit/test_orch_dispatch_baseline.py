"""ORCH-003: dispatch baseline_partitions writes FS-only partition artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
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
            order_regimes=("CF", "IF", "RO"),
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
        ),
    )


@pytest.mark.unit
def test_dispatch__baseline_partitions__writes_fs_only_partition_artifact_holdout_sealed(
    tmp_path: Path,
) -> None:
    """ORCH-003: baseline scores smoke IDs, writes artifact, holdout sealed."""
    from epistemic_sycophancy.runner.cli import dispatch_stage
    from epistemic_sycophancy.runner.identity import clear_stack_cache

    clear_stack_cache()
    artifact_dir = tmp_path / "artifacts"
    study = _study(artifact_dir=str(artifact_dir))
    scored: list[str] = []

    def score_fn(question_ids):
        scored.extend(list(question_ids))
        return {qid: (1.0 if qid == "q1" else -0.5) for qid in question_ids}

    result = dispatch_stage(
        "baseline_partitions",
        study=study,
        freeze_status="unsealed",
        score_fn=score_fn,
    )

    assert result.ok is True
    assert result.stage == "baseline_partitions"
    assert scored == ["q1", "q2"]
    assert "study_fp=" not in result.message or "q_plus" in result.metrics
    assert result.metrics.get("n_q_plus", 0) >= 1
    assert result.metrics.get("n_q_minus", 0) >= 1
    assert "partition" in result.artifacts
    partition_path = Path(result.artifacts["partition"])
    assert partition_path.is_file()
    assert "baseline" in partition_path.parts

    # Holdout remains sealed for this stage path.
    with pytest.raises(HoldoutAccessError):
        dispatch_stage(
            "baseline_partitions",
            study=study,
            freeze_status="unsealed",
            score_fn=score_fn,
            split_name_override="holdout_test_behavior",
        )
