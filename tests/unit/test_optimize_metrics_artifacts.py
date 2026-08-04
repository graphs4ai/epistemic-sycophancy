"""ORCH-LOG-CSV-002: optimize dispatch writes steps/iterations CSV + curves + static."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyFsCoverageConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
)
from epistemic_sycophancy.logging.optimize_metrics import (
    ITERATION_CSV_COLUMNS,
    ITERATION_PLOT_METRICS,
    STEP_CSV_COLUMNS,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.objective.total import ObjectiveResult
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _adam_study(artifact_dir: str, *, coefficient_length: int = 3) -> StudyConfig:
    feature_ids = tuple((17, i + 1) for i in range(coefficient_length))
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
            feature_ids=feature_ids,
            feature_scales=tuple(1.0 for _ in feature_ids),
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
            fs_coverage=StudyFsCoverageConfig(question_ids=("q1",)),
            optimizer=StudyOptimizerConfig(
                kind="projected_adam",
                adam_lr=10.0,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_eps=1e-8,
                adam_microbatch_questions=1,
            ),
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=2,
                question_ids=("qo1",),
            ),
        ),
    )


@pytest.mark.unit
def test_optimize__writes_step_iteration_csv_curves_and_static(
    tmp_path: Path,
) -> None:
    """ORCH-LOG-CSV-002: Adam optimize writes steps/iterations CSV, curves, static.json."""
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

    art = tmp_path / "art"
    study = _adam_study(str(art), coefficient_length=3)

    def objective_fn(beta, question_ids):
        del question_ids
        n_lo = sum(1 for b in beta if b == study.experiment.beta_lower)
        return ObjectiveResult(
            l_resist=0.5 + 0.1 * n_lo,
            l_recover=0.4,
            l_behavior=0.45,
            l_neutral=0.0,
            l_correct=0.0,
            l_beta=0.01 * abs(sum(beta)),
            l_total=0.46 + 0.1 * n_lo,
        )

    def grad_fn(beta, question_ids):
        del question_ids
        # Push first two coords down toward lower bound.
        return tuple(1.0 if i < 2 else 0.0 for i in range(len(beta)))

    result = run_optimize_dispatch(
        study=study,
        freeze_status="unsealed",
        identity_passed=True,
        optimization_question_ids=("qo1",),
        objective_fn=objective_fn,
        grad_fn=grad_fn,
        beta_init=(0.0, 0.0, 0.0),
        n_q_plus=5,
        n_q_minus=3,
    )

    artifacts = result["artifacts"]
    assert "steps_csv" in artifacts
    assert "iterations_csv" in artifacts
    assert "static" in artifacts
    assert "curves_dir" in artifacts

    steps_path = Path(artifacts["steps_csv"])
    iters_path = Path(artifacts["iterations_csv"])
    static_path = Path(artifacts["static"])
    curves_dir = Path(artifacts["curves_dir"])

    assert steps_path.is_file()
    assert iters_path.is_file()
    assert static_path.is_file()
    assert curves_dir.is_dir()

    with steps_path.open(newline="", encoding="utf-8") as handle:
        step_reader = csv.DictReader(handle)
        assert step_reader.fieldnames == list(STEP_CSV_COLUMNS)
        step_rows = list(step_reader)
    assert len(step_rows) == 2
    for row in step_rows:
        assert row["optimizer_kind"] == "projected_adam"
        assert float(row["step_grad_norm"]) == pytest.approx(math.sqrt(2.0), abs=1e-12)
        assert row["l_resist"] != ""
        assert row["l_total"] != ""
        assert int(row["number_at_lower_bound"]) >= 0
        assert int(row["number_at_upper_bound"]) >= 0

    with iters_path.open(newline="", encoding="utf-8") as handle:
        iter_reader = csv.DictReader(handle)
        assert iter_reader.fieldnames == list(ITERATION_CSV_COLUMNS)
        assert "step_grad_norm" not in iter_reader.fieldnames
        iter_rows = list(iter_reader)
    assert len(iter_rows) == 2
    assert iter_rows[0]["l_total"] == step_rows[0]["l_total"]
    assert iter_rows[0]["number_at_lower_bound"] == step_rows[0]["number_at_lower_bound"]

    static = json.loads(static_path.read_text(encoding="utf-8"))
    assert static == {"n_q_plus": 5, "n_q_minus": 3}

    for metric in ITERATION_PLOT_METRICS:
        png = curves_dir / f"{metric}.png"
        assert png.is_file()
        assert png.stat().st_size > 0
