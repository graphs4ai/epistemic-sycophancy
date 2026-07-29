"""ORCH-024: build_objective_fn + build_grad_fn (DEC-076)."""

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
from epistemic_sycophancy.objective.total import evaluate_objective
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
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_adapters__build_objective_and_grad__live_stack_matches_evaluate_objective(
    tmp_path: Path,
) -> None:
    """ORCH-024: objective_fn matches evaluate_objective; grad length = m."""
    from epistemic_sycophancy.runner.adapters.objective import (
        build_grad_fn,
        build_objective_fn,
    )

    study = _study(artifact_dir=str(tmp_path / "art"))
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}

    def margin_scorer(*, belief_condition, question_ids, beta):
        del beta
        if belief_condition == "N":
            return {qid: 1.0 if qid == "q1" else -0.5 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.25,) for qid in question_ids}
        return {qid: (0.75,) for qid in question_ids}

    objective_fn = build_objective_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    beta = (-0.5,)
    eligible = ("q1", "q2")
    loss = objective_fn(beta, eligible)
    assert loss == pytest.approx(
        evaluate_objective(
            ib_margins_by_question={"q1": (0.25,), "q2": (0.25,)},
            cb_margins_by_question={"q1": (0.75,), "q2": (0.75,)},
            baseline_cb_margins={"q1": (0.75,), "q2": (0.75,)},
            baseline_neutral_margins={"q1": 1.0, "q2": -0.5},
            current_neutral_margins={"q1": 1.0, "q2": -0.5},
            q_plus=partitions["q_plus"],
            q_minus=partitions["q_minus"],
            beta=beta,
            tau=1.0,
            w_r=0.5,
            w_u=0.5,
            delta_n=0.0,
            delta_c=0.0,
            lambda_n=0.0,
            lambda_c=0.0,
            lambda_beta=0.01,
        ).l_total
    )
    grad = grad_fn(beta, eligible)
    assert len(grad) == study.experiment.coefficient_length
    assert all(isinstance(x, float) for x in grad)
