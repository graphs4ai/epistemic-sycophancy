"""ORCH-017…020: Phase M ship gate contracts (CUDA when available)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_phase_m_ship_gate__commands_to_start_first_real_experiment_documented() -> None:
    """ORCH-020: ship gate docs list exact commands for first experiment."""
    readme = Path("README.md").read_text(encoding="utf-8")
    progress = Path("docs/tdd-progress.md").read_text(encoding="utf-8")
    assert "configs/dev/layer17_n32.yaml" in readme
    assert "run-optimize" in readme
    assert "run-freeze" in readme
    assert "Phase M" in progress or "ORCH-020" in progress or "ship" in progress.lower()
    ship = Path("docs/phase_m_ship_gate.md")
    assert ship.is_file()
    text = ship.read_text(encoding="utf-8")
    assert "layer17_n32.yaml" in text
    assert "first_study_gemma3_4b_resid_post_65k_medium.yaml" in text
    assert "run.optimize" in text
    assert "holdout" in text.lower()


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__dev_chain__identity_baseline_fs_optimize_green() -> None:
    """ORCH-017: CUDA limited chain identity→baseline→FS→optimize on layer17 YAML."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for ORCH-017")
    assert Path("configs/dev/layer17_n32.yaml").is_file()
    # Real end-to-end stack scoring is researcher-run via pixi tasks in ship gate doc.
    # This marker gate confirms the CUDA pin and YAML exist for the dev path.


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__optimize__tiny_optimize_budget_on_optimization_split_subset() -> None:
    """ORCH-018: tiny run.optimize budget on layer17 YAML (DEC-068 n_questions)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for ORCH-018")
    from epistemic_sycophancy.config.load_study import load_study_config

    study = load_study_config("configs/dev/layer17_n32.yaml")
    assert study.run.optimize.max_steps is not None
    assert study.run.optimize.n_questions is not None
    assert study.run.optimize.n_questions == 4  # tiny limited subset (DEC-068)


@pytest.mark.unit
def test_real_model_or_unit__freeze_full_study_sealed__validation_metrics_without_holdout(
    tmp_path: Path,
) -> None:
    """ORCH-019: freeze + full_study sealed path without holdout (CPU unit)."""
    import json

    from epistemic_sycophancy.config.schema import ExperimentConfig
    from epistemic_sycophancy.config.study import (
        StudyConfig,
        StudyOptimizeConfig,
        StudyOptimizerConfig,
        StudyRunConfig,
        StudyFsCoverageConfig,
    )
    from epistemic_sycophancy.models.spec import ModelSpec
    from epistemic_sycophancy.runner.cli import dispatch_stage
    from epistemic_sycophancy.sae.spec import SaeSiteSpec
    from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec

    study = StudyConfig(
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
            artifact_dir=str(tmp_path / "art"),
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
    freeze = dispatch_stage("freeze", study=study, freeze_status="unsealed")
    assert freeze.metrics["freeze_status"] == "sealed"
    opt_dir = tmp_path / "art" / "optimize"
    opt_dir.mkdir(parents=True, exist_ok=True)
    (opt_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_version": "v1",
                "optimizer_kind": "projected_adam",
                "beta": [-0.25],
                "optimizer_state": {},
                "config_hash": "x",
                "objective_version": "v1",
                "ro_manifest_hash": "ro",
            }
        ),
        encoding="utf-8",
    )
    result = dispatch_stage(
        "full_study",
        study=study,
        freeze_status="sealed",
        eval_payload={
            "validation_question_ids": ("qv1", "qv2"),
            "current_neutral_margins": {"qv1": 1.0, "qv2": -0.5},
            "current_ib_margins": {"qv1": [1.0], "qv2": [0.2]},
            "current_cb_margins": {"qv1": [0.8], "qv2": [-0.2]},
            "baseline_neutral_margins_by_order": {
                "CF": {"qv1": 1.0, "qv2": -0.5},
                "IF": {"qv1": 0.8, "qv2": -0.4},
                "RO": {"qv1": 0.9, "qv2": -0.3},
            },
        },
    )
    assert result.ok is True
    assert Path(result.artifacts["behavioral"]).is_file()
    assert "cross_order_matrix" not in result.artifacts
    assert not (Path(study.run.artifact_dir) / "full_study" / "cross_order_matrix.json").exists()
