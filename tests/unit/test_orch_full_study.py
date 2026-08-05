"""ORCH-014: full_study sealed eval writes single-order behavioral; no holdout."""

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
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
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
def test_dispatch__full_study_sealed__behavioral_no_holdout(
    tmp_path: Path,
) -> None:
    """ORCH-014 / DEC-098: sealed full_study writes intervened + non-intervened logs."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    art = tmp_path / "art"
    study = _study(str(art))
    # Seed best checkpoint from a prior optimize.
    opt_dir = art / "optimize"
    opt_dir.mkdir(parents=True)
    (opt_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_version": "v1",
                "optimizer_kind": "projected_adam",
                "beta": [-0.5],
                "optimizer_state": {},
                "config_hash": "abc",
                "objective_version": "v1",
                "ro_manifest_hash": "ro",
            }
        ),
        encoding="utf-8",
    )

    eval_payload = {
        "validation_question_ids": ("qv1", "qv2"),
        "current_neutral_margins": {"qv1": 1.0, "qv2": -0.5},
        "current_ib_margins": {"qv1": [1.0], "qv2": [0.2]},
        "current_cb_margins": {"qv1": [0.8], "qv2": [-0.2]},
        "baseline_neutral_margins_by_order": {
            "CF": {"qv1": 1.0, "qv2": -0.5},
        },
        # Distinct from intervened so comparison log is not a copy (DEC-098).
        "non_intervened_neutral_margins": {"qv1": 1.0, "qv2": -0.5},
        "non_intervened_ib_margins": {"qv1": [-1.0], "qv2": [-0.8]},
        "non_intervened_cb_margins": {"qv1": [0.1], "qv2": [0.9]},
    }

    with pytest.raises(HoldoutAccessError):
        dispatch_stage(
            "full_study",
            study=study,
            freeze_status="unsealed",
            eval_payload=eval_payload,
        )

    result = dispatch_stage(
        "full_study",
        study=study,
        freeze_status="sealed",
        eval_payload=eval_payload,
        holdout_question_ids=("qh1",),
    )
    assert result.ok is True
    assert "behavioral" in result.artifacts
    assert "behavioral_non_intervened" in result.artifacts
    assert "cross_order_matrix" not in result.artifacts
    behavioral = json.loads(Path(result.artifacts["behavioral"]).read_text())
    assert behavioral["order_regime"] == "CF"
    assert behavioral["beta"] == [-0.5]
    assert "ftw" in behavioral or "selectivity" in behavioral
    non_intervened = json.loads(
        Path(result.artifacts["behavioral_non_intervened"]).read_text()
    )
    assert non_intervened["order_regime"] == "CF"
    assert non_intervened["beta"] == [0.0]
    assert non_intervened["split"] == "behavior_validation"
    for key in ("ftw", "cbr", "selectivity", "pra_mean", "pra_all", "neutral_accuracy"):
        assert key in non_intervened
        assert key in behavioral
    # Same frozen partition denominators; metrics differ under distinct margins.
    assert non_intervened["n_q_plus"] == behavioral["n_q_plus"]
    assert non_intervened["n_q_minus"] == behavioral["n_q_minus"]
    assert non_intervened["ftw"] != behavioral["ftw"]
    assert non_intervened["cbr"] != behavioral["cbr"]
    assert (art / "full_study" / "behavioral_non_intervened.json").is_file()
    assert not (art / "full_study" / "cross_order_matrix.json").exists()
    # Holdout IDs must not appear in artifacts.
    text = Path(result.artifacts["behavioral"]).read_text()
    assert "qh1" not in text
    assert "qh1" not in Path(result.artifacts["behavioral_non_intervened"]).read_text()


def _write_ckpt(path: Path, beta: list[float], *, best_by: str) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint_version": "v1",
                "optimizer_kind": "projected_adam",
                "beta": beta,
                "optimizer_state": {"best_by": best_by, "index": 0, "value": 0.1},
                "config_hash": "abc",
                "objective_version": "v1",
                "ro_manifest_hash": "ro",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_dispatch__full_study_sealed__writes_behavioral_best_by_criterion(
    tmp_path: Path,
) -> None:
    """ORCH-014c / DEC-100: per-criterion best-β behavioral logs on validation."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    art = tmp_path / "art"
    study = _study(str(art))
    opt_dir = art / "optimize"
    opt_dir.mkdir(parents=True)
    _write_ckpt(opt_dir / "best_checkpoint.json", [-0.5], best_by="l_total")
    _write_ckpt(opt_dir / "best_checkpoint_by_l_total.json", [-0.5], best_by="l_total")
    _write_ckpt(opt_dir / "best_checkpoint_by_l_resist.json", [-1.25], best_by="l_resist")

    total_margins = {
        "neutral": {"qv1": 1.0, "qv2": -0.5},
        "ib": {"qv1": [1.0], "qv2": [0.2]},
        "cb": {"qv1": [0.8], "qv2": [-0.2]},
        "beta": [-0.5],
    }
    # IB on Q+ flips negative so FTW differs from the l_total intervened log.
    resist_margins = {
        "neutral": {"qv1": 0.8, "qv2": -0.2},
        "ib": {"qv1": [-1.5], "qv2": [0.9]},
        "cb": {"qv1": [0.4], "qv2": [0.6]},
        "beta": [-1.25],
    }
    eval_payload = {
        "validation_question_ids": ("qv1", "qv2"),
        "current_neutral_margins": total_margins["neutral"],
        "current_ib_margins": total_margins["ib"],
        "current_cb_margins": total_margins["cb"],
        "baseline_neutral_margins_by_order": {"CF": {"qv1": 1.0, "qv2": -0.5}},
        "non_intervened_neutral_margins": {"qv1": 1.0, "qv2": -0.5},
        "non_intervened_ib_margins": {"qv1": [-1.0], "qv2": [-0.8]},
        "non_intervened_cb_margins": {"qv1": [0.1], "qv2": [0.9]},
        "margins_by_criterion": {
            "l_total": total_margins,
            "l_resist": resist_margins,
        },
    }

    result = dispatch_stage(
        "full_study",
        study=study,
        freeze_status="sealed",
        eval_payload=eval_payload,
        holdout_question_ids=("qh1",),
    )
    assert result.ok is True
    assert "behavioral" in result.artifacts
    assert "behavioral_non_intervened" in result.artifacts
    assert "behavioral_best_by_l_total" in result.artifacts
    assert "behavioral_best_by_l_resist" in result.artifacts

    by_total = json.loads(
        Path(result.artifacts["behavioral_best_by_l_total"]).read_text()
    )
    by_resist = json.loads(
        Path(result.artifacts["behavioral_best_by_l_resist"]).read_text()
    )
    legacy = json.loads(Path(result.artifacts["behavioral"]).read_text())
    assert by_total["beta"] == [-0.5]
    assert by_resist["beta"] == [-1.25]
    assert by_total["selection_criterion"] == "l_total"
    assert by_resist["selection_criterion"] == "l_resist"
    assert by_total["selection_split"] == "optimization"
    assert by_resist["selection_split"] == "optimization"
    assert legacy["beta"] == by_total["beta"]
    assert legacy["ftw"] == by_total["ftw"]
    assert by_resist["ftw"] != by_total["ftw"]
    assert (art / "full_study" / "behavioral_best_by_l_resist.json").is_file()
    assert "qh1" not in Path(result.artifacts["behavioral_best_by_l_resist"]).read_text()
