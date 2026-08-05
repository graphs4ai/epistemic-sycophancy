"""OPT-MULTI-001: optimize writes best_checkpoint_by_{metric}.json (DEC-100)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyFsCoverageConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.objective.total import ObjectiveResult
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec

_CRITERIA = (
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
)


def _study(artifact_dir: str, *, max_steps: int = 3) -> StudyConfig:
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
                adam_lr=1.0,
                adam_beta1=0.0,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=max_steps,
                question_ids=("qo1",),
            ),
        ),
    )


@pytest.mark.unit
def test_optimize__multi_criterion_opt_split__writes_best_checkpoint_by_metric(
    tmp_path: Path,
) -> None:
    """OPT-MULTI-001: one best_checkpoint_by_{metric}.json per DEC-097 loss key."""
    from epistemic_sycophancy.optimization.checkpoint import load_checkpoint
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

    art = tmp_path / "art"
    study = _study(str(art), max_steps=3)

    # Hand-picked trajectory: distinct per-component minima at known steps.
    # step0: recover wins; step1: resist + beta win; step2: behavior/neutral/correct/total win.
    trajectory = [
        ObjectiveResult(
            l_resist=1.0,
            l_recover=0.1,
            l_behavior=0.5,
            l_neutral=0.9,
            l_correct=0.8,
            l_beta=0.7,
            l_total=0.5,
        ),
        ObjectiveResult(
            l_resist=0.1,
            l_recover=1.0,
            l_behavior=0.4,
            l_neutral=0.2,
            l_correct=0.3,
            l_beta=0.05,
            l_total=0.9,
        ),
        ObjectiveResult(
            l_resist=0.5,
            l_recover=0.5,
            l_behavior=0.1,
            l_neutral=0.1,
            l_correct=0.1,
            l_beta=0.5,
            l_total=0.2,
        ),
    ]
    expected_winner_index = {
        "l_resist": 1,
        "l_recover": 0,
        "l_behavior": 2,
        "l_neutral": 2,
        "l_correct": 2,
        "l_beta": 1,
        "l_total": 2,
    }
    logged_betas: list[list[float]] = []
    call = {"n": 0}

    def objective_fn(beta, question_ids):
        del question_ids
        i = call["n"]
        call["n"] += 1
        logged_betas.append([float(x) for x in beta])
        return trajectory[i]

    def grad_fn(beta, question_ids):
        del beta, question_ids
        return (1.0,)

    result = run_optimize_dispatch(
        study=study,
        freeze_status="unsealed",
        identity_passed=True,
        optimization_question_ids=("qo1",),
        objective_fn=objective_fn,
        grad_fn=grad_fn,
        beta_init=(0.0,),
    )

    assert len(logged_betas) == 3
    out_dir = art / "optimize"
    best_path = Path(result["artifacts"]["best_checkpoint"])
    assert best_path == out_dir / "best_checkpoint.json"
    best = load_checkpoint(json.loads(best_path.read_text(encoding="utf-8")))
    assert best["beta"] == logged_betas[expected_winner_index["l_total"]]

    for metric in _CRITERIA:
        path = out_dir / f"best_checkpoint_by_{metric}.json"
        assert path.is_file(), f"missing {path.name}"
        artifact_key = f"best_checkpoint_by_{metric}"
        assert artifact_key in result["artifacts"]
        assert Path(result["artifacts"][artifact_key]) == path
        ckpt = load_checkpoint(json.loads(path.read_text(encoding="utf-8")))
        winner = expected_winner_index[metric]
        assert ckpt["beta"] == logged_betas[winner]
        state = ckpt["optimizer_state"]
        assert state["best_by"] == metric
        assert state["index"] == winner
        assert state["value"] == pytest.approx(
            float(getattr(trajectory[winner], metric)), abs=1e-12
        )

    # Legacy alias matches l_total criterion file.
    by_total = load_checkpoint(
        json.loads((out_dir / "best_checkpoint_by_l_total.json").read_text())
    )
    assert best["beta"] == by_total["beta"]
