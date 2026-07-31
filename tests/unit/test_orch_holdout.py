"""ORCH-015: holdout_eval requires sealed freeze + mark_holdout_started."""

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
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__holdout_eval__requires_sealed_freeze_and_mark_holdout_started(
    tmp_path: Path,
) -> None:
    """ORCH-015: holdout_eval unlocks only after sealed + mark_holdout_started."""
    from epistemic_sycophancy.runner.cli import dispatch_stage
    from epistemic_sycophancy.runner.freeze_stage import run_freeze_dispatch

    study = _study(str(tmp_path / "art"))
    freeze = run_freeze_dispatch(study=study)
    frozen_path = Path(freeze["artifacts"]["frozen_config"])

    with pytest.raises(HoldoutAccessError):
        dispatch_stage(
            "holdout_eval",
            study=study,
            freeze_status="unsealed",
            frozen_config_path=str(frozen_path),
            holdout_rows_provider=lambda: [{"question_id": "qh1"}],
        )

    # Sealed but holdout_started still False on disk until holdout_eval runs.
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert payload["holdout_started"] is False

    result = dispatch_stage(
        "holdout_eval",
        study=study,
        freeze_status="sealed",
        frozen_config_path=str(frozen_path),
        holdout_rows_provider=lambda: [{"question_id": "qh1", "split": "holdout"}],
    )
    assert result.ok is True
    assert result.metrics.get("holdout_started") is True
    assert result.metrics.get("n_holdout_rows") == 1
    updated = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert updated["holdout_started"] is True
    assert "holdout" in result.artifacts
