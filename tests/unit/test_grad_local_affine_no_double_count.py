"""GRAD-013: local affine at live M(β₀) must use δ, not double-count β₀."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.config.schema import ExperimentConfig
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


def _study(*, artifact_dir: str, **exp_overrides: object) -> StudyConfig:
    exp_kwargs: dict[str, object] = {
        "tau": 1.0,
        "lambda_n": 0.0,
        "lambda_c": 0.0,
        "lambda_beta": 0.0,
        "delta_n": 0.1,
        "delta_c": 0.0,
        "w_r": 1.0,
        "w_u": 0.0,
        "beta_lower": -2.0,
        "beta_upper": 0.0,
        "feature_ids": ((17, 1),),
        "feature_scales": (1.0,),
        "coefficient_length": 1,
        "tie_policy": "merge_into_q_minus",
        "tie_band_epsilon": 1e-6,
        "mc1_tie_policy": "fail_and_report",
        "invalid_row_policy": "fail_trial",
        "multi_token_candidate_scoring": "sum_log_probs",
        "ro_manifest_selection": "primary_single",
        "continuation_A": "A",
        "continuation_B": "B",
        "continuation_include_eos": False,
        "attribution_scope": "last_prompt_token",
        "pool_eligibility_override": False,
        "pool_quota_per_list": 8,
    }
    exp_kwargs.update(exp_overrides)
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
        experiment=ExperimentConfig(**exp_kwargs),
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
def test_adapters__build_grad_fn__nonzero_beta__local_affine_matches_live_margin(
    tmp_path: Path,
) -> None:
    """GRAD-013a: at β₀≠0, surrogate uses M(β₀)+J·δ (δ=0), not M(β₀)+J·β₀.

    Hand chain rule for resistance-only φ(M)=softplus(-M/τ) with τ=1:
      M_live=0.5, J=2, β₀=-1
      ∂φ/∂M = -σ(-0.5)
      ∂L/∂β = -σ(-0.5)·2
    Double-counting would evaluate at M=-1.5 and yield -σ(1.5)·2.
    """
    from epistemic_sycophancy.runner.adapters.objective import build_grad_fn

    study = _study(artifact_dir=str(tmp_path / "art"))
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    beta0 = (-1.0,)
    m_live_ib = 0.5
    jac_val = 2.0
    j = torch.tensor([jac_val], dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64)

    def margin_scorer(*, belief_condition, question_ids, beta):
        del beta
        if belief_condition == "N":
            return {qid: 1.0 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (m_live_ib,) for qid in question_ids}
        return {qid: (0.75,) for qid in question_ids}

    def margin_jacobian_fn(*, beta, question_ids, partitions):
        del beta, question_ids, partitions
        return {
            "ib_margin_jac": {"q1": [j.clone()], "q2": [z.clone()]},
            "cb_margin_jac": {"q1": [z.clone()], "q2": [z.clone()]},
            "neutral_margin_jac": {"q1": z.clone(), "q2": z.clone()},
        }

    sigmoid_neg_half = 1.0 / (1.0 + math.exp(0.5))
    expected_grad = -sigmoid_neg_half * jac_val
    buggy_m = m_live_ib + jac_val * beta0[0]
    assert buggy_m == pytest.approx(-1.5)
    sigmoid_pos_one_half = 1.0 / (1.0 + math.exp(-1.5))
    buggy_grad = -sigmoid_pos_one_half * jac_val
    assert expected_grad != pytest.approx(buggy_grad, abs=1e-6)

    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
        margin_jacobian_fn=margin_jacobian_fn,
    )
    grad = grad_fn(beta0, ("q1", "q2"))
    assert len(grad) == 1
    assert float(grad[0]) == pytest.approx(expected_grad, abs=1e-8, rel=1e-6)
    assert float(grad[0]) != pytest.approx(buggy_grad, abs=1e-6)


@pytest.mark.unit
def test_adapters__build_grad_fn__nonzero_beta__hinge_uses_live_margin_not_double_count(
    tmp_path: Path,
) -> None:
    """GRAD-013b: soft-hinge local affine must use live M, not double-count β₀.

    Neutral soft-hinge (DEC-101): δ_n=0.1, baseline=1.0, M_live=0.95, τ=1,
    excess=-0.05. With two identical questions and J=1:
      ∂d/∂M = -σ(-0.05), L_neutral mean ⇒ ∂L/∂β = -σ(-0.05).
    Double-counting evaluates at buggy M=0.45 (excess=+0.45) ⇒ -σ(0.45).
    """
    from epistemic_sycophancy.objective.total import (
        evaluate_objective_with_grad,
        evaluate_objective_with_local_grad,
    )
    from epistemic_sycophancy.runner.adapters.objective import build_grad_fn

    study = _study(
        artifact_dir=str(tmp_path / "art"),
        lambda_n=1.0,
        lambda_beta=0.0,
    )
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    beta0 = (-0.5,)
    m_baseline_n = 1.0
    m_live_n = 0.95
    jac_val = 1.0
    j = torch.tensor([jac_val], dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64)
    beta_t = torch.tensor(list(beta0), dtype=torch.float64)

    def margin_scorer(*, belief_condition, question_ids, beta):
        at_zero = all(float(b) == 0.0 for b in beta)
        if belief_condition == "N":
            value = m_baseline_n if at_zero else m_live_n
            return {qid: value for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.0,) for qid in question_ids}
        return {qid: (0.0,) for qid in question_ids}

    def margin_jacobian_fn(*, beta, question_ids, partitions):
        del beta, question_ids, partitions
        return {
            "ib_margin_jac": {"q1": [z.clone()], "q2": [z.clone()]},
            "cb_margin_jac": {"q1": [z.clone()], "q2": [z.clone()]},
            "neutral_margin_jac": {"q1": j.clone(), "q2": j.clone()},
        }

    live_excess = m_baseline_n - m_live_n - 0.1
    assert live_excess == pytest.approx(-0.05)
    buggy_m = m_live_n + jac_val * beta0[0]
    buggy_excess = m_baseline_n - buggy_m - 0.1
    assert buggy_excess == pytest.approx(0.45)

    expected_local = -1.0 / (1.0 + math.exp(-live_excess))
    expected_buggy = -1.0 / (1.0 + math.exp(-buggy_excess))
    assert expected_local != pytest.approx(expected_buggy, abs=1e-6)

    live_n = {"q1": m_live_n, "q2": m_live_n}
    baseline_n = {"q1": m_baseline_n, "q2": m_baseline_n}
    ib_live = {"q1": (0.0,), "q2": (0.0,)}
    cb_live = {"q1": (0.0,), "q2": (0.0,)}
    jac_payload = margin_jacobian_fn(
        beta=beta0, question_ids=("q1", "q2"), partitions=partitions
    )
    common = dict(
        ib_margin_jac=jac_payload["ib_margin_jac"],
        cb_margin_jac=jac_payload["cb_margin_jac"],
        baseline_cb_margins=cb_live,
        baseline_neutral_margins=baseline_n,
        neutral_margin_jac=jac_payload["neutral_margin_jac"],
        q_plus=partitions["q_plus"],
        q_minus=partitions["q_minus"],
        tau=1.0,
        w_r=1.0,
        w_u=0.0,
        delta_n=0.1,
        delta_c=0.0,
        lambda_n=1.0,
        lambda_c=0.0,
        lambda_beta=0.0,
    )
    _, local_grad = evaluate_objective_with_local_grad(
        beta=beta_t,
        ib_margins_live=ib_live,
        cb_margins_live=cb_live,
        neutral_margins_live=live_n,
        **common,
    )
    assert float(local_grad[0]) == pytest.approx(expected_local, abs=1e-8, rel=1e-6)

    _, buggy_grad = evaluate_objective_with_grad(
        beta=beta_t,
        ib_margin_const=ib_live,
        cb_margin_const=cb_live,
        neutral_margin_const=live_n,
        **common,
    )
    assert float(buggy_grad[0]) == pytest.approx(expected_buggy, abs=1e-8, rel=1e-6)

    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
        margin_jacobian_fn=margin_jacobian_fn,
    )
    grad = grad_fn(beta0, ("q1", "q2"))
    assert float(grad[0]) == pytest.approx(expected_local, abs=1e-8, rel=1e-6)
