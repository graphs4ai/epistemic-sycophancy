"""ORCH-LOG: operational pipeline logging (DEC-089)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudyFsCoverageConfig,
)
from epistemic_sycophancy.logging.pipeline import (
    PIPELINE_LOGGER_NAME,
    configure_pipeline_logging,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.runner.cli import build_arg_parser, run_cli
from epistemic_sycophancy.runner.identity import clear_stack_cache
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "adapters"


class _Tok:
    def __call__(self, texts, return_tensors="pt", padding=True):
        batch = len(texts)
        return {
            "input_ids": torch.zeros(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"A": [0], "B": [1]}[text]


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        # Row counter (not batch index) so prompt_batch_size=1 still alternates.
        self._row = 0

    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 3, dtype=torch.float64)
        # Alternate A-favoring / B-favoring so Q+/Q- are both nonempty under CF.
        for i in range(batch):
            if self._row % 2 == 0:
                logits[i, -1, :] = torch.tensor([2.0, -1.0, 0.0])
            else:
                logits[i, -1, :] = torch.tensor([-1.0, 2.0, 0.0])
            self._row += 1
        return SimpleNamespace(logits=logits)


class _FakeStack:
    def __init__(self) -> None:
        self.model = _ToyCausalLM()
        self.tokenizer = _Tok()
        self.device = torch.device("cpu")


def _write_tiny_study_yaml(path: Path, artifact_dir: Path) -> None:
    payload = {
        "stack": {
            "model": {
                "hf_id": "google/gemma-3-4b-it",
                "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
                "tokenizer_revision": "093f9f388b31de276ce2de164bdc2081324b9767",
                "dtype": "bfloat16",
                "device_policy": "cuda_required",
            },
            "sae": {
                "release": "gemma-scope-2-4b-it-res",
                "site": "resid_post",
                "width": "width_65k",
                "l0": "l0_medium",
                "layers": [17],
            },
            "hooks": {
                "token_scope": "last_prompt_token",
                "resolver_id": "gemma3_resid_post",
                "k": None,
            },
        },
        "experiment": {
            "tau": 1.0,
            "lambda_n": 0.0,
            "lambda_c": 0.0,
            "lambda_beta": 0.01,
            "delta_n": 0.0,
            "delta_c": 0.0,
            "w_r": 0.5,
            "w_u": 0.5,
            "beta_lower": -2.0,
            "beta_upper": 0.0,
            "feature_ids": [],
            "feature_scales": [],
            "coefficient_length": 0,
            "tie_policy": "merge_into_q_minus",
            "tie_band_epsilon": 1.0e-6,
            "mc1_tie_policy": "fail_and_report",
            "invalid_row_policy": "fail_trial",
            "multi_token_candidate_scoring": "sum_log_probs",
            "ro_manifest_selection": "primary_single",
            "continuation_A": "A",
            "continuation_B": "B",
            "continuation_include_eos": False,
            "attribution_scope": "last_prompt_token",
            "pool_eligibility_override": False,
            "pool_quota_per_list": 8,
        },
        "run": {
            "artifact_dir": str(artifact_dir),
            "order_regime": "CF",
            "feature_chunk_size": 1024,
            "prompt_batch_size": 1,
            "fs_coverage": {"question_ids": ["q_fs_1", "q_fs_2"]},
            "optimizer": {
                "kind": "projected_adam",
                "adam_lr": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1.0e-8,
                "adam_microbatch_questions": 1,
            },
            "optimize": {
                "budget_match_on": "n_objective_evals",
                "max_steps": 2,
                "n_questions": 2,
            },
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _optimize_study(artifact_dir: str) -> StudyConfig:
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
                question_ids=("qo1",),
            ),
        ),
    )


@pytest.mark.unit
def test_pipeline_logging__configure__attaches_stderr_handler_at_requested_level() -> None:
    """ORCH-LOG-001: configure_pipeline_logging sets package logger level + stderr handler."""
    root = logging.getLogger(PIPELINE_LOGGER_NAME)
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    configure_pipeline_logging(level="INFO")

    assert root.level == logging.INFO
    assert any(
        isinstance(h, logging.StreamHandler) and h.level == logging.INFO
        for h in root.handlers
    )


@pytest.mark.unit
def test_cli__arg_parser__exposes_log_level_flag() -> None:
    """ORCH-LOG-001: CLI accepts --log-level (default INFO)."""
    parser = build_arg_parser()
    args = parser.parse_args(
        ["identity", "--config", "x.yaml", "--log-level", "DEBUG"]
    )
    assert args.log_level == "DEBUG"
    defaults = parser.parse_args(["identity", "--config", "x.yaml"])
    assert defaults.log_level == "INFO"


@pytest.mark.unit
def test_cli__run_cli__emits_stage_start_and_end_with_elapsed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ORCH-LOG-002: run_cli logs stage start/end with elapsed_s around dispatch."""
    clear_stack_cache()
    cfg = tmp_path / "study.yaml"
    art = tmp_path / "art"
    _write_tiny_study_yaml(cfg, art)
    configure_pipeline_logging(level="INFO")
    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER_NAME):
        code = run_cli(
            ["baseline_partitions", "--config", str(cfg), "--log-level", "INFO"],
            stack_loader=lambda _study: _FakeStack(),
            corpus_jsonl_paths=(FIXTURE_ROOT / "processed_mc0_tiny.jsonl",),
            split_manifest_path=FIXTURE_ROOT / "split_manifest_tiny.csv",
        )
    assert code == 0
    messages = [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER_NAME]
    assert any("stage=baseline_partitions starting" in m for m in messages)
    end = next(m for m in messages if "stage=baseline_partitions completed" in m)
    assert "elapsed_s=" in end
    assert "ok=True" in end


@pytest.mark.unit
def test_optimize__progress__logs_each_step_with_l_total(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ORCH-LOG-003: optimize emits per-step progress with trial_index and l_total."""
    from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

    study = _optimize_study(str(tmp_path / "art"))
    # Configure before capture so propagate=False logger is attached to caplog
    # (pytest misses loggers that become non-propagating after fixture enter).
    configure_pipeline_logging(level="INFO")
    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    pipeline_logger.addHandler(caplog.handler)

    def objective_fn(beta, question_ids):
        del question_ids
        return float(beta[0])

    def grad_fn(beta, question_ids):
        del beta, question_ids
        return (1.0,)

    try:
        with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER_NAME):
            run_optimize_dispatch(
                study=study,
                freeze_status="unsealed",
                identity_passed=True,
                optimization_question_ids=("qo1",),
                objective_fn=objective_fn,
                grad_fn=grad_fn,
                beta_init=(0.0,),
            )
    finally:
        pipeline_logger.removeHandler(caplog.handler)
    messages = [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER_NAME]
    steps = [m for m in messages if "progress=optimize_step" in m]
    assert len(steps) == 2
    assert "trial_index=0" in steps[0] and "l_total=" in steps[0]
    assert "trial_index=1" in steps[1]


@pytest.mark.unit
def test_fs_dispatch__progress__logs_component_and_skips(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ORCH-LOG-004: FS logs each component start and surfaces empty-component skips."""
    from epistemic_sycophancy.runner.fs_dispatch import run_feature_selection_dispatch

    study = _optimize_study(str(tmp_path / "art"))
    # Shrink pool quota so empty recovery still builds from resistance.
    object.__setattr__(study.experiment, "pool_quota_per_list", 1)
    configure_pipeline_logging(level="INFO")

    def jacobian_fn(*, order_regime, question_ids, component):
        del order_regime, question_ids
        if component == "resistance":
            return {(17, 1): 1.5}
        return {}

    def scale_fn(keys):
        return {k: 1.0 for k in keys}

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER_NAME):
        run_feature_selection_dispatch(
            study=study,
            freeze_status="unsealed",
            jacobian_fn=jacobian_fn,
            scale_fn=scale_fn,
            question_ids=("q1",),
        )
    messages = [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER_NAME]
    assert any("progress=fs_component" in m and "component=resistance" in m for m in messages)
    assert any(
        "progress=fs_component_skip" in m and "component=recovery" in m for m in messages
    )


@pytest.mark.unit
def test_baseline__progress__logs_partition_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ORCH-LOG-006: baseline emits progress with Q+/Q- counts for the study order."""
    from epistemic_sycophancy.runner.baseline import run_baseline_dispatch

    study = _optimize_study(str(tmp_path / "art"))
    configure_pipeline_logging(level="INFO")

    def score_fn(question_ids):
        # Force one Q+ and one Q- under ε=1e-6.
        return {qid: (1.0 if i == 0 else -1.0) for i, qid in enumerate(question_ids)}

    with caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER_NAME):
        run_baseline_dispatch(
            study=study,
            freeze_status="unsealed",
            score_fn=score_fn,
            question_ids=("q1", "q2"),
        )
    messages = [r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER_NAME]
    assert any(
        "progress=baseline_partition" in m
        and "order_regime=CF" in m
        and "n_q_plus=" in m
        and "n_q_minus=" in m
        for m in messages
    )


@pytest.mark.unit
def test_freeze_and_holdout__audit__emit_warning_events(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ORCH-LOG-005: freeze seal and holdout unseal emit WARNING audit lines."""
    from epistemic_sycophancy.runner.freeze_stage import run_freeze_dispatch
    from epistemic_sycophancy.runner.holdout_eval import run_holdout_eval_dispatch

    study = _optimize_study(str(tmp_path / "art"))
    configure_pipeline_logging(level="INFO")

    with caplog.at_level(logging.WARNING, logger=PIPELINE_LOGGER_NAME):
        frozen = run_freeze_dispatch(study=study)
        run_holdout_eval_dispatch(
            study=study,
            freeze_status="sealed",
            frozen_config_path=frozen["artifacts"]["frozen_config"],
            holdout_rows_provider=lambda: [{"question_id": "qh1"}],
        )
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == PIPELINE_LOGGER_NAME and r.levelno >= logging.WARNING
    ]
    assert any("audit=freeze_sealed" in m for m in warnings)
    assert any("audit=holdout_unsealed" in m for m in warnings)
