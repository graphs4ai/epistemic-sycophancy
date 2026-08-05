"""OPT-ES-001: projected Adam early stopping via run.optimize.patience (DEC-099)."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig, InvalidExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyFsCoverageConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study(*, artifact_dir: str, max_steps: int, patience: int | None) -> StudyConfig:
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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1",)),
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
                max_steps=max_steps,
                patience=patience,
                question_ids=("qo1",),
            ),
        ),
    )


@pytest.mark.unit
def test_optimize_config__patience__must_be_positive_int_when_set() -> None:
    """OPT-ES-001a: patience omit/None ok; non-positive rejected; Adam-only."""
    ok = StudyOptimizeConfig(
        budget_match_on="n_objective_evals",
        max_steps=10,
        patience=3,
    )
    assert ok.patience == 3
    omitted = StudyOptimizeConfig(
        budget_match_on="n_objective_evals",
        max_steps=10,
    )
    assert omitted.patience is None

    with pytest.raises(InvalidExperimentConfig, match="patience"):
        StudyOptimizeConfig(
            budget_match_on="n_objective_evals",
            max_steps=10,
            patience=0,
        )
    with pytest.raises(InvalidExperimentConfig, match="patience"):
        StudyOptimizeConfig(
            budget_match_on="n_objective_evals",
            n_trials=5,
            population_size=4,
            patience=2,
        )


@pytest.mark.unit
def test_optimize__projected_adam_patience__stops_after_stale_steps_without_improvement(
    tmp_path: Path,
) -> None:
    """OPT-ES-001: stop after ``patience`` consecutive non-improving opt-split steps.

    Loss schedule (post-step): 5 → 4 (improve) → 4.5 → 4.6 (two stales with
    patience=2) → would continue to 4.7… if max_steps were honored alone.
    """
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

    losses = [5.0, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2]
    call_i = {"n": 0}

    def objective_fn(beta, qids):  # noqa: ANN001
        del beta, qids
        idx = min(call_i["n"], len(losses) - 1)
        call_i["n"] += 1
        return float(losses[idx])

    result = run_optimize_dispatch(
        study=_study(artifact_dir=str(tmp_path / "art"), max_steps=10, patience=2),
        freeze_status="unsealed",
        identity_passed=True,
        optimization_question_ids=("qo1",),
        objective_fn=objective_fn,
        grad_fn=lambda beta, qids: tuple(0.0 for _ in beta),
    )
    assert result["metrics"]["n_trials"] == 4
    assert result["metrics"]["stopped_early"] is True
    assert result["metrics"]["patience"] == 2
    assert result["metrics"]["best_l_total"] == pytest.approx(4.0)
    assert call_i["n"] == 4
