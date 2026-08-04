"""ADAPT-OBJ-001: build_objective_fn returns ObjectiveResult; coerce float|result."""

from __future__ import annotations

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
from epistemic_sycophancy.objective.total import ObjectiveResult, evaluate_objective
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
                max_steps=1,
                question_ids=("q1", "q2"),
            ),
        ),
    )


@pytest.mark.unit
def test_build_objective_fn__returns_objective_result_with_components(
    tmp_path: Path,
) -> None:
    """ADAPT-OBJ-001: production objective_fn returns ObjectiveResult, not bare float."""
    from epistemic_sycophancy.runner.adapters.objective import build_objective_fn

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
    result = objective_fn((-0.5,), ("q1", "q2"))
    assert isinstance(result, ObjectiveResult)
    expected = evaluate_objective(
        ib_margins_by_question={"q1": (0.25,), "q2": (0.25,)},
        cb_margins_by_question={"q1": (0.75,), "q2": (0.75,)},
        baseline_cb_margins={"q1": (0.75,), "q2": (0.75,)},
        baseline_neutral_margins={"q1": 1.0, "q2": -0.5},
        current_neutral_margins={"q1": 1.0, "q2": -0.5},
        q_plus=partitions["q_plus"],
        q_minus=partitions["q_minus"],
        beta=(-0.5,),
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.0,
        delta_c=0.0,
        lambda_n=0.0,
        lambda_c=0.0,
        lambda_beta=0.01,
    )
    assert result.l_total == pytest.approx(expected.l_total, abs=1e-12, rel=1e-12)
    assert result.l_resist == pytest.approx(expected.l_resist, abs=1e-12, rel=1e-12)
    assert result.l_recover == pytest.approx(expected.l_recover, abs=1e-12, rel=1e-12)
    assert result.l_behavior == pytest.approx(expected.l_behavior, abs=1e-12, rel=1e-12)
    assert result.l_neutral == pytest.approx(expected.l_neutral, abs=1e-12, rel=1e-12)
    assert result.l_correct == pytest.approx(expected.l_correct, abs=1e-12, rel=1e-12)
    assert result.l_beta == pytest.approx(expected.l_beta, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_coerce_objective__float_result_and_mapping__normalize_components() -> None:
    """ADAPT-OBJ-001b: coerce accepts float, ObjectiveResult, or mapping."""
    from epistemic_sycophancy.runner.optimize import coerce_objective

    loss, comps = coerce_objective(1.25)
    assert loss == 1.25
    assert comps["l_total"] == 1.25
    assert comps["l_resist"] is None

    result = ObjectiveResult(
        l_resist=0.1,
        l_recover=0.2,
        l_behavior=0.15,
        l_neutral=0.0,
        l_correct=0.0,
        l_beta=0.01,
        l_total=0.16,
    )
    loss2, comps2 = coerce_objective(result)
    assert loss2 == 0.16
    assert comps2["l_resist"] == 0.1
    assert comps2["l_total"] == 0.16

    loss3, comps3 = coerce_objective(
        {
            "l_resist": 0.3,
            "l_recover": 0.4,
            "l_behavior": 0.35,
            "l_neutral": 0.0,
            "l_correct": 0.0,
            "l_beta": 0.0,
            "l_total": 0.35,
        }
    )
    assert loss3 == 0.35
    assert comps3["l_recover"] == 0.4
