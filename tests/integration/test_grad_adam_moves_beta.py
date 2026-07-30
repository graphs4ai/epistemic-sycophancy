"""GRAD-007: projected Adam via production build_grad_fn moves β from 0."""

from __future__ import annotations

import json
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
            lambda_beta=0.0,
            delta_n=0.0,
            delta_c=0.0,
            w_r=0.5,
            w_u=0.5,
            beta_lower=-2.0,
            beta_upper=0.0,
            feature_ids=((17, 0), (17, 2)),
            feature_scales=(2.0, 0.5),
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
                max_steps=5,
                question_ids=("q1", "q2"),
            ),
        ),
    )


@pytest.mark.integration
def test_optimize__build_grad_fn_projected_adam__moves_beta_from_zero(
    tmp_path: Path,
) -> None:
    """GRAD-007: production build_grad_fn + Adam changes β within bounds."""
    import importlib.util
    from pathlib import Path as P

    from epistemic_sycophancy.runner.adapters.objective import (
        build_grad_fn,
        build_objective_fn,
    )
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

    toy_path = (
        P(__file__).resolve().parents[1]
        / "fixtures"
        / "feature_selection"
        / "toy_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("toy_gradients_grad007", toy_path)
    assert spec is not None and spec.loader is not None
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)

    study = _study(str(tmp_path / "art"))
    decoder = toy.spec_decoder()
    latents = toy.spec_latents()
    scales = toy.spec_scales()
    g = -toy.spec_gradient()  # flip so ∂L/∂β > 0 → Adam suppresses (β↓ from 0)

    class _FakeStack:
        def margin_projection_batch(
            self,
            *,
            belief_condition: str,
            question_ids: tuple[str, ...],
            beta: tuple[float, ...],
        ):
            del beta
            n = len(question_ids)
            return {
                "layer": 17,
                "residual_gradients": g.unsqueeze(0).expand(n, -1).clone(),
                "latents": latents.unsqueeze(0).expand(n, -1).clone(),
                "decoder": decoder,
                "feature_scales": scales,
                "question_ids": list(question_ids),
                "belief_condition": belief_condition,
            }

    def margin_scorer(*, belief_condition, question_ids, beta):
        # Affine-ish live margins: M ≈ M0 + J_selected · β using FEAT-004 slice.
        # Const margins at β=0; scorer ignores β (const) — jac carries ∂M/∂β.
        del beta
        if belief_condition == "N":
            return {qid: 0.0 for qid in question_ids}
        if belief_condition == "IB":
            return {qid: (0.0,) for qid in question_ids}
        return {qid: (0.0,) for qid in question_ids}

    partitions = {"q_plus": frozenset({"q1"}), "q_minus": frozenset({"q2"})}
    stack = _FakeStack()
    objective_fn = build_objective_fn(
        study,
        stack,
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    grad_fn = build_grad_fn(
        study,
        stack,
        partitions=partitions,
        margin_scorer=margin_scorer,
    )
    # Sanity: grad at β=0 is informative before optimize.
    g0 = grad_fn((0.0, 0.0), ("q1", "q2"))
    assert math.sqrt(sum(x * x for x in g0)) > 0.0

    run_optimize_dispatch(
        study=study,
        freeze_status="unsealed",
        identity_passed=True,
        optimization_question_ids=("q1", "q2"),
        objective_fn=objective_fn,
        grad_fn=grad_fn,
        beta_init=(0.0, 0.0),
    )
    trials_path = Path(study.run.artifact_dir) / "optimize" / "trials.jsonl"
    rows = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 5
    final_beta = [float(x) for x in rows[-1]["beta"]]
    assert any(abs(b) > 1e-8 for b in final_beta), final_beta
    assert all(study.experiment.beta_lower <= b <= study.experiment.beta_upper for b in final_beta)
    losses = [float(r["l_total"]) for r in rows]
    assert losses[-1] != losses[0] or any(abs(b) > 1e-8 for b in final_beta)
