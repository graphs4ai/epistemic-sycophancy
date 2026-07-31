"""FSC-006: pool artifact schema v2 with nominator provenance; reject stale v1."""

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
    StudyFsCoverageConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.runner.adapters.pool import load_common_pool_artifact
from epistemic_sycophancy.runner.fs_dispatch import run_feature_selection_dispatch
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
            pool_quota_per_list=2,
        ),
        run=StudyRunConfig(
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=8,
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
                max_steps=1,
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_fs_pool__schema_v2__records_nominator_provenance(tmp_path: Path) -> None:
    """FSC-006 / DEC-085: each selected feature records nominating lists + signed J."""
    study = _study(str(tmp_path / "art"))

    def jacobian_fn(*, order_regime: str, question_ids, component: str):
        del order_regime, question_ids
        if component == "resistance":
            return {(17, 1): 5.0, (17, 2): 4.0}
        if component == "recovery":
            return {(17, 2): 3.0, (17, 3): 2.0}
        if component == "neutral_surrogate":
            return {(17, 1): -0.5, (17, 2): 0.1, (17, 3): 0.2}
        return {(17, 1): 0.7, (17, 2): -0.2, (17, 3): 0.9}

    result = run_feature_selection_dispatch(
        study=study,
        freeze_status="unsealed",
        jacobian_fn=jacobian_fn,
        scale_fn=lambda keys: {k: 1.0 for k in keys},
        question_ids=("q1", "q2"),
        optimization_question_ids=("q_opt",),
    )
    path = Path(result["artifacts"]["pool"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert "provenance" in payload
    # Feature (17,2) nominated by both resistance and recovery.
    prov_2 = payload["provenance"]["17:2"]
    nominators = {(n["order"], n["component"]) for n in prov_2["nominators"]}
    assert ("CF", "resistance") in nominators
    assert ("CF", "recovery") in nominators
    assert "surrogates" in prov_2
    assert prov_2["surrogates"]["neutral_surrogate"] == pytest.approx(0.1)
    assert prov_2["surrogates"]["correct_surrogate"] == pytest.approx(-0.2)

    loaded = load_common_pool_artifact(path)
    assert set(loaded.feature_ids) == {(17, 1), (17, 2), (17, 3)}


@pytest.mark.unit
def test_fs_pool__stale_v1_artifact__rejected_on_load(tmp_path: Path) -> None:
    """FSC-006: optimize/load rejects neutral-only v1 pools (force re-run-fs)."""
    path = tmp_path / "common_pool.json"
    path.write_text(
        json.dumps(
            {
                "feature_ids": [[17, 1]],
                "feature_scales": [1.0],
                "pool_size": 1,
                "scale_source": "decoder_norm",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version|stale|re-run"):
        load_common_pool_artifact(path)
