"""ORCH-031: full_study builds eval_payload from best checkpoint + validation."""

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
from epistemic_sycophancy.optimization.checkpoint import dump_checkpoint
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
                max_steps=2,
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__full_study__builds_eval_payload_from_best_checkpoint_validation(
    tmp_path: Path,
) -> None:
    """ORCH-031: eval_payload=None → build from stack + best β; no holdout."""
    clear_stack_cache()
    art = tmp_path / "art"
    study = _study(artifact_dir=str(art))
    opt = art / "optimize"
    opt.mkdir(parents=True)
    ckpt = dump_checkpoint(
        optimizer_kind="projected_adam",
        beta=[-0.25],
        optimizer_state={},
        config_hash="cfg",
        objective_version="v1",
        ro_manifest_hash="ro",
    )
    (opt / "best_checkpoint.json").write_text(
        json.dumps(ckpt) + "\n", encoding="utf-8"
    )

    class _Stack:
        def score_belief_margins(
            self, *, belief_condition, question_ids, beta, order_regime="CF"
        ):
            del beta, order_regime
            if belief_condition == "N":
                return {qid: 0.5 if qid.endswith("1") else -0.5 for qid in question_ids}
            if belief_condition == "IB":
                return {qid: 0.1 for qid in question_ids}
            return {qid: 0.9 for qid in question_ids}

    result = dispatch_stage(
        "full_study",
        study=study,
        freeze_status="sealed",
        stack_loader=lambda _s: _Stack(),
        eval_payload=None,
        validation_question_ids=("q_val_1", "q_val_2"),
        holdout_question_ids=("q_hold_1",),
    )
    assert result.ok
    assert (art / "full_study" / "behavioral.json").is_file()
    assert (art / "full_study" / "behavioral_non_intervened.json").is_file()
    behavioral = json.loads(
        (art / "full_study" / "behavioral.json").read_text(encoding="utf-8")
    )
    non_intervened = json.loads(
        (art / "full_study" / "behavioral_non_intervened.json").read_text(
            encoding="utf-8"
        )
    )
    assert "q_hold_1" not in str(behavioral)
    assert "q_hold_1" not in str(non_intervened)
    assert non_intervened["beta"] == [0.0]
    assert behavioral["beta"] == [-0.25]


@pytest.mark.unit
def test_dispatch__full_study__multi_criterion_checkpoints__scores_each_beta(
    tmp_path: Path,
) -> None:
    """ORCH-031c / DEC-100: default adapter scores each best_checkpoint_by_* β."""
    clear_stack_cache()
    art = tmp_path / "art"
    study = _study(artifact_dir=str(art))
    opt = art / "optimize"
    opt.mkdir(parents=True)

    def _dump(beta: list[float], *, best_by: str) -> dict:
        return dump_checkpoint(
            optimizer_kind="projected_adam",
            beta=beta,
            optimizer_state={"best_by": best_by, "index": 0, "value": 0.1},
            config_hash="cfg",
            objective_version="v1",
            ro_manifest_hash="ro",
        )

    (opt / "best_checkpoint.json").write_text(
        json.dumps(_dump([-0.25], best_by="l_total")) + "\n", encoding="utf-8"
    )
    (opt / "best_checkpoint_by_l_total.json").write_text(
        json.dumps(_dump([-0.25], best_by="l_total")) + "\n", encoding="utf-8"
    )
    (opt / "best_checkpoint_by_l_resist.json").write_text(
        json.dumps(_dump([-1.5], best_by="l_resist")) + "\n", encoding="utf-8"
    )

    scored_betas: list[tuple[float, ...]] = []

    class _Stack:
        def score_belief_margins(
            self, *, belief_condition, question_ids, beta, order_regime="CF"
        ):
            del order_regime
            beta_t = tuple(float(x) for x in beta)
            if belief_condition == "N":
                scored_betas.append(beta_t)
            scale = 1.0 + abs(sum(beta_t))
            if belief_condition == "N":
                return {
                    qid: (0.5 if qid.endswith("1") else -0.5) * scale
                    for qid in question_ids
                }
            if belief_condition == "IB":
                # Flip Q+ under the resist β so FTW differs.
                if beta_t == (-1.5,):
                    return {qid: -1.0 for qid in question_ids}
                return {qid: 0.1 for qid in question_ids}
            return {qid: 0.9 for qid in question_ids}

    result = dispatch_stage(
        "full_study",
        study=study,
        freeze_status="sealed",
        stack_loader=lambda _s: _Stack(),
        eval_payload=None,
        validation_question_ids=("q_val_1", "q_val_2"),
        holdout_question_ids=("q_hold_1",),
    )
    assert result.ok
    assert "behavioral_best_by_l_total" in result.artifacts
    assert "behavioral_best_by_l_resist" in result.artifacts
    by_total = json.loads(
        Path(result.artifacts["behavioral_best_by_l_total"]).read_text(encoding="utf-8")
    )
    by_resist = json.loads(
        Path(result.artifacts["behavioral_best_by_l_resist"]).read_text(
            encoding="utf-8"
        )
    )
    assert by_total["beta"] == [-0.25]
    assert by_resist["beta"] == [-1.5]
    assert by_total["selection_criterion"] == "l_total"
    assert by_resist["selection_criterion"] == "l_resist"
    assert by_resist["ftw"] != by_total["ftw"]
    assert set(scored_betas) == {(0.0,), (-0.25,), (-1.5,)}
    assert "q_hold_1" not in str(by_resist)
