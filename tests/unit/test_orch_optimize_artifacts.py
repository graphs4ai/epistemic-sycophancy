"""ORCH-011: optimize writes TrialRecords and best checkpoint artifact."""

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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1")),
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
                question_ids=("qo1",),
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__optimize__writes_trial_records_and_best_checkpoint_artifact(
    tmp_path: Path,
) -> None:
    """ORCH-011: trials.jsonl + best_checkpoint.json + loss_curve.png + metrics CSV (DEC-032/091/097)."""
    from epistemic_sycophancy.optimization.checkpoint import load_checkpoint
    from epistemic_sycophancy.runner.cli import dispatch_stage

    result = dispatch_stage(
        "optimize",
        study=_study(str(tmp_path / "art")),
        freeze_status="unsealed",
        identity_passed=True,
        objective_fn=lambda beta, qids: float(sum(beta)),
        grad_fn=lambda beta, qids: tuple(0.1 for _ in beta),
        optimization_question_ids=("qo1",),
    )
    assert "trials" in result.artifacts
    assert "best_checkpoint" in result.artifacts
    assert "loss_curve" in result.artifacts
    assert "steps_csv" in result.artifacts
    assert "iterations_csv" in result.artifacts
    trials_path = Path(result.artifacts["trials"])
    best_path = Path(result.artifacts["best_checkpoint"])
    curve_path = Path(result.artifacts["loss_curve"])
    steps_path = Path(result.artifacts["steps_csv"])
    iters_path = Path(result.artifacts["iterations_csv"])
    assert trials_path.is_file()
    assert best_path.is_file()
    assert curve_path.is_file()
    assert curve_path.stat().st_size > 0
    assert curve_path.name == "loss_curve.png"
    assert steps_path.is_file()
    assert iters_path.is_file()
    assert steps_path.name == "steps.csv"
    assert iters_path.name == "iterations.csv"
    lines = [json.loads(line) for line in trials_path.read_text().splitlines() if line]
    assert len(lines) == 2
    for row in lines:
        assert "trial_index" in row
        assert "beta" in row
        assert "l_total" in row
        assert "optimizer_kind" in row
    ckpt = load_checkpoint(json.loads(best_path.read_text()))
    assert ckpt["checkpoint_version"] == "v1"
    assert ckpt["optimizer_kind"] == "projected_adam"
    assert len(ckpt["beta"]) == 1
