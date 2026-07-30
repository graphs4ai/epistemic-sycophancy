"""ORCH-030: optimize builds objective/grad from stack + run.optimize budget."""

from __future__ import annotations

import json
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
from epistemic_sycophancy.runner.cli import dispatch_stage
from epistemic_sycophancy.runner.identity import clear_stack_cache
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
            feature_ids=(),
            feature_scales=(),
            coefficient_length=0,
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
                max_steps=2,
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__optimize__builds_objective_grad_from_stack_and_run_optimize_budget(
    tmp_path: Path,
) -> None:
    """ORCH-030: None objective/grad/identity → adapters; uses run.optimize.max_steps."""
    clear_stack_cache()
    art = tmp_path / "art"
    study = _study(artifact_dir=str(art))
    (art / "identity").mkdir(parents=True)
    (art / "identity" / "identity_result.json").write_text(
        json.dumps({"identity_passed": True, "max_abs_diff": 0.0}) + "\n",
        encoding="utf-8",
    )
    (art / "feature_selection").mkdir(parents=True)
    (art / "feature_selection" / "common_pool.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "feature_ids": [[17, 1]],
                "feature_scales": [1.0],
                "pool_size": 1,
                "scale_source": "decoder_norm",
                "provenance": {
                    "17:1": {
                        "nominators": [
                            {
                                "order": "CF",
                                "component": "resistance",
                                "signed_jacobian": 1.0,
                            }
                        ],
                        "surrogates": {},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (art / "baseline").mkdir(parents=True)
    (art / "baseline" / "partition_CF.json").write_text(
        json.dumps({"q_plus": ["q1"], "q_minus": ["q2"], "q_tie": []}) + "\n",
        encoding="utf-8",
    )

    # Tiny linear SAE batch: pool feature (17,1) active with nonzero ∂M/∂x so
    # real coefficient_jacobian yields informative ∂M/∂β (GRAD-010 / DEC-084).
    decoder = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    latents_row = torch.tensor([0.0, 1.0], dtype=torch.float64)  # feature 1 active
    scales = torch.tensor([1.0, 1.0], dtype=torch.float64)
    residual_g = torch.tensor([0.0, 1.0], dtype=torch.float64)

    class _Stack:
        def score_belief_margins(self, *, belief_condition, question_ids, beta):
            del beta
            if belief_condition == "N":
                return {qid: 1.0 if qid == "q1" else -0.5 for qid in question_ids}
            if belief_condition == "IB":
                return {qid: (0.25,) for qid in question_ids}
            return {qid: (0.75,) for qid in question_ids}

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
                "residual_gradients": residual_g.unsqueeze(0).expand(n, -1).clone(),
                "latents": latents_row.unsqueeze(0).expand(n, -1).clone(),
                "decoder": decoder,
                "feature_scales": scales,
                "question_ids": list(question_ids),
                "belief_condition": belief_condition,
            }

    result = dispatch_stage(
        "optimize",
        study=study,
        freeze_status="unsealed",
        stack_loader=lambda _s: _Stack(),
        objective_fn=None,
        grad_fn=None,
        identity_passed=None,
        optimization_question_ids=("q1", "q2", "q3", "q4"),
    )
    assert result.ok
    assert result.metrics["n_trials"] == 2  # run.optimize.max_steps, not smoke 1
    ckpt = Path(result.artifacts["best_checkpoint"])
    assert ckpt.is_file()
