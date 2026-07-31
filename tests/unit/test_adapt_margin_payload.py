"""ORCH-023: build_margin_payload for optimize / optimize (DEC-076)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1", "q2")),
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
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_adapters__build_margin_payload__contains_required_optimize_keys(
    tmp_path: Path,
) -> None:
    """ORCH-023: margin_payload has optimize keys; live scorer called at given beta."""
    from epistemic_sycophancy.runner.adapters.margins import build_margin_payload

    study = _study(artifact_dir=str(tmp_path / "art"))
    stack = object()
    calls: list[tuple[float, ...]] = []

    def margin_scorer(
        *,
        belief_condition: str,
        question_ids: Sequence[str],
        beta: Sequence[float],
    ) -> Mapping[str, Any]:
        calls.append(tuple(float(b) for b in beta))
        # N → scalar; IB/CB → variant sequences.
        if belief_condition == "N":
            return {qid: 1.0 if qid == "q1" else -0.5 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.25,) for qid in question_ids}
        if belief_condition == "CB":
            return {qid: (0.75,) for qid in question_ids}
        raise ValueError(belief_condition)

    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    beta = (-0.5,)
    payload = build_margin_payload(
        study,
        stack,
        beta=beta,
        question_ids=("q1", "q2"),
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    required = {
        "ib_margins_by_question",
        "cb_margins_by_question",
        "baseline_cb_margins",
        "baseline_neutral_margins",
        "current_neutral_margins",
        "q_plus",
        "q_minus",
    }
    assert required <= set(payload)
    assert payload["q_plus"] == frozenset({"q1"})
    assert payload["q_minus"] == frozenset({"q2"})
    assert payload["current_neutral_margins"]["q1"] == pytest.approx(1.0)
    assert payload["ib_margins_by_question"]["q1"] == (0.25,)
    assert payload["cb_margins_by_question"]["q2"] == (0.75,)
    # Live scoring: current margins scored at the requested beta (DEC-076).
    assert beta in calls
    assert calls.count(beta) >= 3  # N + IB + CB at current beta

