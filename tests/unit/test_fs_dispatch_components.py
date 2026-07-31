"""FSC-005: fs_dispatch emits four distinct canonical component Jacobian maps."""

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
from epistemic_sycophancy.feature_selection.components import COMPONENT_CONDITION
from epistemic_sycophancy.models.spec import ModelSpec
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
def test_fs_dispatch__four_distinct_component_maps__canonical_names(
    tmp_path: Path,
) -> None:
    """FSC-005 / DEC-085: no replication; names match components.py."""
    study = _study(str(tmp_path / "art"))
    calls: list[str] = []

    def jacobian_fn(*, order_regime: str, question_ids, component: str):
        del order_regime, question_ids
        calls.append(component)
        # Distinct positive signed J per component so pool sees different lists.
        offsets = {
            "resistance": 10,
            "recovery": 20,
            "neutral_surrogate": 30,
            "correct_surrogate": 40,
        }
        base = offsets[component]
        return {
            (17, base): 3.0,
            (17, base + 1): 2.0,
            (17, base + 2): -1.0,  # nonpositive: not nominated
        }

    def scale_fn(keys):
        return {k: 1.0 for k in keys}

    result = run_feature_selection_dispatch(
        study=study,
        freeze_status="unsealed",
        jacobian_fn=jacobian_fn,
        scale_fn=scale_fn,
        question_ids=("q1", "q2"),
        optimization_question_ids=("q_opt",),
        validation_question_ids=("q_val",),
        holdout_question_ids=("q_hold",),
    )
    assert set(calls) == set(COMPONENT_CONDITION)
    assert len(calls) == 4
    # Drifted names must not appear.
    assert "neutral_preservation" not in calls
    assert "correct_belief" not in calls

    pool = result["pool"]
    # DEC-019: only resistance/recovery nominate → features 10,11,20,21.
    assert set(pool.feature_ids) == {(17, 10), (17, 11), (17, 20), (17, 21)}
    # Surrogate-only features must not enter the pool.
    assert (17, 30) not in pool.feature_ids
    assert (17, 40) not in pool.feature_ids

    # Per-component maps for the study order only (DEC-087).
    component_maps = result["component_jacobians"]
    for component in COMPONENT_CONDITION:
        assert ("CF", component) in component_maps
    assert component_maps[("CF", "resistance")] != component_maps[("CF", "recovery")]
    assert (
        component_maps[("CF", "neutral_surrogate")]
        != component_maps[("CF", "correct_surrogate")]
    )
    assert component_maps[("CF", "resistance")][(17, 10)] == 3.0
    assert ("IF", "resistance") not in component_maps

    path = Path(result["artifacts"]["pool"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pool_size"] == 4
