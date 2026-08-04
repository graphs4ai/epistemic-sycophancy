"""ORCH-025: build_eval_payload for full_study (DEC-069)."""

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
                max_steps=20,
                n_questions=1,
            ),
        ),
    )


@pytest.mark.unit
def test_adapters__build_eval_payload__validation_margins_for_full_study(
    tmp_path: Path,
) -> None:
    """ORCH-025: eval_payload has full_study keys; no holdout IDs."""
    from epistemic_sycophancy.runner.adapters.eval_payload import build_eval_payload

    study = _study(artifact_dir=str(tmp_path / "art"))
    validation_ids = ("q_val_1", "q_val_2")
    holdout_ids = ("q_hold_1",)

    def margin_scorer(*, belief_condition, question_ids, beta, order_regime="CF"):
        del beta, order_regime
        if any(q in holdout_ids for q in question_ids):
            raise HoldoutAccessError("holdout leaked into eval scorer")
        if belief_condition == "N":
            return {qid: 0.5 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: 0.1 for qid in question_ids}
        return {qid: 0.9 for qid in question_ids}

    payload = build_eval_payload(
        study,
        stack=object(),
        best_beta=(-0.25,),
        validation_question_ids=validation_ids,
        margin_scorer=margin_scorer,
        holdout_question_ids=holdout_ids,
    )
    required = {
        "current_neutral_margins",
        "current_ib_margins",
        "current_cb_margins",
        "baseline_neutral_margins_by_order",
        "non_intervened_neutral_margins",
        "non_intervened_ib_margins",
        "non_intervened_cb_margins",
    }
    assert required <= set(payload)
    assert set(payload["current_neutral_margins"]) == set(validation_ids)
    assert set(payload["non_intervened_neutral_margins"]) == set(validation_ids)
    assert set(payload["non_intervened_ib_margins"]) == set(validation_ids)
    assert set(payload["non_intervened_cb_margins"]) == set(validation_ids)
    assert "q_hold_1" not in payload["current_neutral_margins"]
    assert "q_hold_1" not in payload["non_intervened_neutral_margins"]
    assert set(payload["baseline_neutral_margins_by_order"]) == {"CF"}
    assert payload["order_regime"] == "CF"
    # β=0 N margins feed both baseline partition and non-intervened Acc_N.
    assert (
        payload["non_intervened_neutral_margins"]
        == payload["baseline_neutral_margins_by_order"]["CF"]
    )
    with pytest.raises(HoldoutAccessError):
        build_eval_payload(
            study,
            stack=object(),
            best_beta=(-0.25,),
            validation_question_ids=holdout_ids,
            margin_scorer=margin_scorer,
            holdout_question_ids=holdout_ids,
        )
