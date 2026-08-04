"""GRAD-004: build_grad_fn defaults to projected margin jac; loud zero-grad."""

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
    StudyFsCoverageConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
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
            feature_ids=tuple((17, i) for i in range(m)),
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
def test_adapters__build_grad_fn__without_jac_source__raises(
    tmp_path: Path,
) -> None:
    """GRAD-004: no silent all-zero jac default when stack cannot supply ∂M/∂β."""
    from epistemic_sycophancy.runner.adapters.objective import build_grad_fn

    study = _study(artifact_dir=str(tmp_path / "art"))
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}

    def margin_scorer(*, belief_condition, question_ids, beta):
        del beta
        if belief_condition == "N":
            return {qid: 1.0 if qid == "q1" else -0.5 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.0,) for qid in question_ids}
        return {qid: (0.5,) for qid in question_ids}

    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    with pytest.raises(ValueError, match="margin_jacobian|∂M/∂β|margin projection"):
        grad_fn((0.0, 0.0), ("q1", "q2"))


@pytest.mark.unit
def test_adapters__build_grad_fn__stack_margin_projection__nonzero_grad(
    tmp_path: Path,
) -> None:
    """GRAD-004: default path projects stack margin grads via coefficient_jacobian."""
    from epistemic_sycophancy.runner.adapters.objective import build_grad_fn

    import importlib.util
    from pathlib import Path as P

    toy_path = (
        P(__file__).resolve().parents[1]
        / "fixtures"
        / "feature_selection"
        / "toy_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("toy_gradients_grad004", toy_path)
    assert spec is not None and spec.loader is not None
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)

    # Pool selects features 0 and 2 from the FEAT-004 3-wide SAE (m=2).
    study = _study(artifact_dir=str(tmp_path / "art"), m=2)
    study.experiment.feature_ids = ((17, 0), (17, 2))
    study.experiment.feature_scales = (2.0, 0.5)
    study.experiment.coefficient_length = 2

    decoder = toy.spec_decoder()
    latents = toy.spec_latents()
    scales = toy.spec_scales()
    g = toy.spec_gradient()

    class _FakeStack:
        def margin_projection_batch(
            self,
            *,
            belief_condition: str,
            question_ids: tuple[str, ...],
            beta: tuple[float, ...],
            layer: int | None = None,
        ):
            _ = layer
            del beta
            # One prompt row per question; residual grad = FEAT-004 g.
            n = len(question_ids)
            return {
                "residual_gradients": g.unsqueeze(0).expand(n, -1).clone(),
                "latents": latents.unsqueeze(0).expand(n, -1).clone(),
                "decoder": decoder,
                "feature_scales": scales,
                "question_ids": list(question_ids),
                "belief_condition": belief_condition,
            }

    def margin_scorer(*, belief_condition, question_ids, beta):
        del beta
        if belief_condition == "N":
            return {qid: 0.0 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.0,) for qid in question_ids}
        return {qid: (0.0,) for qid in question_ids}

    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    grad_fn = build_grad_fn(
        study,
        stack=_FakeStack(),
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    grad = grad_fn((0.0, 0.0), ("q1", "q2"))
    grad_norm = math.sqrt(sum(float(x) * float(x) for x in grad))
    assert grad_norm > 0.0
    assert len(grad) == 2


@pytest.mark.unit
def test_adapters__build_grad_fn__identically_zero_grad__raises(
    tmp_path: Path,
) -> None:
    """GRAD-004 / DEC-084: loud fail when ∂L/∂β is identically zero."""
    from epistemic_sycophancy.runner.adapters.objective import build_grad_fn

    study = _study(artifact_dir=str(tmp_path / "art"), m=2)
    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    zero = torch.zeros(2, dtype=torch.float64)

    def margin_scorer(*, belief_condition, question_ids, beta):
        del beta
        if belief_condition == "N":
            return {qid: 1.0 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (1.0,) for qid in question_ids}
        return {qid: (1.0,) for qid in question_ids}

    def margin_jacobian_fn(*, beta, question_ids, partitions):
        del beta, question_ids, partitions
        return {
            "ib_margin_jac": {"q1": [zero.clone()], "q2": [zero.clone()]},
            "cb_margin_jac": {"q1": [zero.clone()], "q2": [zero.clone()]},
            "neutral_margin_jac": {"q1": zero.clone(), "q2": zero.clone()},
        }

    grad_fn = build_grad_fn(
        study,
        stack=object(),
        partitions=partitions,
        margin_scorer=margin_scorer,
        margin_jacobian_fn=margin_jacobian_fn,
    )
    with pytest.raises(ValueError, match="identically zero|non-finite"):
        grad_fn((0.0, 0.0), ("q1", "q2"))
