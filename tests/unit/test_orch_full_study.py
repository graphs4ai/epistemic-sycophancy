"""ORCH-014: full_study sealed eval writes single-order behavioral; no holdout."""

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
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__full_study_sealed__behavioral_no_holdout(
    tmp_path: Path,
) -> None:
    """ORCH-014: sealed full_study writes single-order val metrics; holdout sealed."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    art = tmp_path / "art"
    study = _study(str(art))
    # Seed best checkpoint from a prior optimize.
    opt_dir = art / "optimize"
    opt_dir.mkdir(parents=True)
    (opt_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_version": "v1",
                "optimizer_kind": "projected_adam",
                "beta": [-0.5],
                "optimizer_state": {},
                "config_hash": "abc",
                "objective_version": "v1",
                "ro_manifest_hash": "ro",
            }
        ),
        encoding="utf-8",
    )

    eval_payload = {
        "validation_question_ids": ("qv1", "qv2"),
        "current_neutral_margins": {"qv1": 1.0, "qv2": -0.5},
        "current_ib_margins": {"qv1": [1.0], "qv2": [0.2]},
        "current_cb_margins": {"qv1": [0.8], "qv2": [-0.2]},
        "baseline_neutral_margins_by_order": {
            "CF": {"qv1": 1.0, "qv2": -0.5},
        },
    }

    with pytest.raises(HoldoutAccessError):
        dispatch_stage(
            "full_study",
            study=study,
            freeze_status="unsealed",
            eval_payload=eval_payload,
        )

    result = dispatch_stage(
        "full_study",
        study=study,
        freeze_status="sealed",
        eval_payload=eval_payload,
        holdout_question_ids=("qh1",),
    )
    assert result.ok is True
    assert "behavioral" in result.artifacts
    assert "cross_order_matrix" not in result.artifacts
    behavioral = json.loads(Path(result.artifacts["behavioral"]).read_text())
    assert behavioral["order_regime"] == "CF"
    assert "ftw" in behavioral or "selectivity" in behavioral
    assert not (art / "full_study" / "cross_order_matrix.json").exists()
    # Holdout IDs must not appear in artifacts.
    text = Path(result.artifacts["behavioral"]).read_text()
    assert "qh1" not in text
