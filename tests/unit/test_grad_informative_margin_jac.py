"""GRAD-002: build_grad_fn must use informative margin Jacobians."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudySmokeConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.objective.total import evaluate_objective_with_grad
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study(*, artifact_dir: str, m: int = 2) -> StudyConfig:
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
            feature_ids=tuple((17, i + 1) for i in range(m)),
            feature_scales=tuple(1.0 for _ in range(m)),
            coefficient_length=m,
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
def test_adapters__build_grad_fn__informative_margin_jac__nonzero_matches_reference(
    tmp_path: Path,
) -> None:
    """GRAD-002: informative ∂M/∂β → ‖grad‖>0 and matches evaluate_objective_with_grad."""
    from epistemic_sycophancy.runner.adapters.objective import build_grad_fn

    m = 2
    study = _study(artifact_dir=str(tmp_path / "art"), m=m)
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}

    # Synthetic affine: M = M0 + J·β with informative nonzero J (hand-built).
    ib_const = {"q1": (0.0,), "q2": (0.0,)}
    cb_const = {"q1": (1.0,), "q2": (-1.0,)}
    neutral_const = {"q1": 1.0, "q2": -1.0}
    ib_j = {
        "q1": [torch.tensor([1.0, 0.0], dtype=torch.float64)],
        "q2": [torch.tensor([0.5, 0.5], dtype=torch.float64)],
    }
    cb_j = {
        "q1": [torch.tensor([0.0, 1.0], dtype=torch.float64)],
        "q2": [torch.tensor([-1.0, 0.0], dtype=torch.float64)],
    }
    neutral_j = {
        "q1": torch.tensor([0.2, -0.3], dtype=torch.float64),
        "q2": torch.tensor([0.1, 0.4], dtype=torch.float64),
    }

    def margin_scorer(*, belief_condition, question_ids, beta):
        del beta
        if belief_condition == "N":
            return {qid: neutral_const[qid] for qid in question_ids}
        if belief_condition == "IB":
            return {qid: ib_const[qid] for qid in question_ids}
        return {qid: cb_const[qid] for qid in question_ids}

    def margin_jacobian_fn(*, beta, question_ids, partitions):
        del beta, question_ids, partitions
        return {
            "ib_margin_jac": {
                qid: [t.clone() for t in rows] for qid, rows in ib_j.items()
            },
            "cb_margin_jac": {
                qid: [t.clone() for t in rows] for qid, rows in cb_j.items()
            },
            "neutral_margin_jac": {
                qid: t.clone() for qid, t in neutral_j.items()
            },
        }

    beta0 = (0.0, 0.0)
    eligible = ("q1", "q2")
    _, expected_grad = evaluate_objective_with_grad(
        beta=torch.tensor(beta0, dtype=torch.float64),
        ib_margin_const=ib_const,
        ib_margin_jac=ib_j,
        cb_margin_const=cb_const,
        cb_margin_jac=cb_j,
        baseline_cb_margins=cb_const,
        baseline_neutral_margins=neutral_const,
        neutral_margin_const=neutral_const,
        neutral_margin_jac=neutral_j,
        q_plus=partitions["q_plus"],
        q_minus=partitions["q_minus"],
        tau=1.0,
        w_r=0.5,
        w_u=0.5,
        delta_n=0.0,
        delta_c=0.0,
        lambda_n=0.0,
        lambda_c=0.0,
        lambda_beta=0.01,
    )
    expected_norm = math.sqrt(sum(g * g for g in expected_grad))
    assert expected_norm > 0.0

    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
        margin_jacobian_fn=margin_jacobian_fn,
    )
    grad = grad_fn(beta0, eligible)
    grad_norm = math.sqrt(sum(float(g) * float(g) for g in grad))
    assert grad_norm > 0.0
    assert len(grad) == m
    for got, exp in zip(grad, expected_grad):
        assert float(got) == pytest.approx(float(exp), abs=1e-8, rel=1e-6)
