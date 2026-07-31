"""ORCH-009: dispatch optimize uses optimization split and YAML budgets."""

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
            lambda_n=0.0,
            lambda_c=0.0,
            lambda_beta=0.01,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=((17, 1), (17, 2)),
            feature_scales=(1.0, 1.0),
            coefficient_length=2,
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
                adam_microbatch_questions=1,  # optimizer hyperparams only
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=3,
                question_ids=("qo1", "qo2", "qo3"),
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__optimize__uses_optimization_split_and_yaml_budgets_not_optimize(
    tmp_path: Path,
) -> None:
    """ORCH-009: optimize uses run.optimize budgets + opt split."""
    from epistemic_sycophancy.runner.cli import STAGE_ORDER, dispatch_stage

    assert "optimize" in STAGE_ORDER
    assert STAGE_ORDER.index("feature_selection") + 1 == STAGE_ORDER.index("optimize")

    study = _study(artifact_dir=str(tmp_path / "art"))
    seen_qids: list[tuple[str, ...]] = []
    seen_steps: list[int] = []

    def objective_fn(beta, question_ids):
        seen_qids.append(tuple(question_ids))
        # Prefer more negative β (suppression).
        return float(sum(beta) ** 2) + 0.1 * len(question_ids)

    def grad_fn(beta, question_ids):
        del question_ids
        seen_steps.append(1)
        return tuple(2.0 * float(b) for b in beta)

    result = dispatch_stage(
        "optimize",
        study=study,
        freeze_status="unsealed",
        identity_passed=True,
        objective_fn=objective_fn,
        grad_fn=grad_fn,
        optimization_question_ids=("qo1", "qo2", "qo3", "qo4"),
    )

    assert result.ok is True
    assert result.stage == "optimize"
    # Used optimize.question_ids, not fs_coverage allowlist alone.
    assert all(qids == ("qo1", "qo2", "qo3") for qids in seen_qids)
    assert len(seen_steps) == 3  # run.optimize.max_steps
    assert "best_checkpoint" in result.artifacts
    ckpt = Path(result.artifacts["best_checkpoint"])
    assert ckpt.is_file()
    beta = result.metrics["best_beta"]
    assert len(beta) == 2
    assert all(study.experiment.beta_lower <= b <= study.experiment.beta_upper for b in beta)
