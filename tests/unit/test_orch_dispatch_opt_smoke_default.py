"""ORCH-029: opt_smoke builds margin_payload/beta/identity from artifacts."""

from __future__ import annotations

import json
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
from epistemic_sycophancy.runner.cli import dispatch_stage
from epistemic_sycophancy.runner.identity import clear_stack_cache
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study(*, artifact_dir: str, coefficient_length: int = 1) -> StudyConfig:
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
            feature_ids=((17, 1),) if coefficient_length else (),
            feature_scales=(1.0,) if coefficient_length else (),
            coefficient_length=coefficient_length,
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
def test_dispatch__opt_smoke__builds_margin_payload_beta_and_identity_from_artifacts(
    tmp_path: Path,
) -> None:
    """ORCH-029: None margin/beta/identity → load artifacts + live scorer."""
    clear_stack_cache()
    art = tmp_path / "art"
    study = _study(artifact_dir=str(art), coefficient_length=0)
    # Identity artifact.
    ident = art / "identity"
    ident.mkdir(parents=True)
    (ident / "identity_result.json").write_text(
        json.dumps({"identity_passed": True, "max_abs_diff": 0.0}) + "\n",
        encoding="utf-8",
    )
    # Pool artifact (DEC-073).
    fs = art / "feature_selection"
    fs.mkdir(parents=True)
    (fs / "common_pool.json").write_text(
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
    # Baseline partition.
    base = art / "baseline"
    base.mkdir(parents=True)
    (base / "partition_CF.json").write_text(
        json.dumps(
            {
                "order_regime": "CF",
                "q_plus": ["q1"],
                "q_minus": ["q2"],
                "q_tie": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _Stack:
        def score_belief_margins(self, *, belief_condition, question_ids, beta):
            del beta
            if belief_condition == "N":
                return {qid: 1.0 if qid == "q1" else -0.5 for qid in question_ids}
            if belief_condition == "IB":
                return {qid: (0.25,) for qid in question_ids}
            return {qid: (0.75,) for qid in question_ids}

    result = dispatch_stage(
        "opt_smoke",
        study=study,
        freeze_status="unsealed",
        stack_loader=lambda _s: _Stack(),
        margin_payload=None,
        beta=None,
        identity_passed=None,
    )
    assert result.ok
    assert "l_total" in result.metrics
    assert result.metrics["l_total"] == result.metrics["l_total"]  # finite
