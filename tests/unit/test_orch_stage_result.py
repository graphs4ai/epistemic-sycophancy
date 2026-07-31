"""ORCH-006: StageResult schema includes hashes and structured fields."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import pytest
import torch

from epistemic_sycophancy.config.schema import ExperimentConfig
from epistemic_sycophancy.config.study import (
    StudyConfig,
    StudyOptimizeConfig,
    StudyOptimizerConfig,
    StudyRunConfig,
    StudyFsCoverageConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec


def _study() -> StudyConfig:
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
            lambda_n=1.0,
            lambda_c=1.0,
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
            artifact_dir="artifacts/dev/layer17_n32",
            order_regime="CF",
            feature_chunk_size=1024,
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
                max_steps=20,
                n_questions=4,
            ),
        ),
    )


class _FakeStack:
    def __init__(self) -> None:
        self.config = _study().stack
        self._residual = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)

    def capture_layer_residuals(self, *, texts, layers):
        del texts
        return {int(layer): self._residual.clone() for layer in layers}

    @contextmanager
    def install_hooks(self, **kwargs) -> Iterator[None]:
        del kwargs
        yield


@pytest.mark.unit
def test_stage_result__schema__includes_ok_artifacts_metrics_and_stack_hashes() -> None:
    """ORCH-006: StageResult carries study_yaml_fingerprint and stack hashes."""
    from epistemic_sycophancy.runner.cli import StageResult, dispatch_stage

    required = {
        "ok",
        "artifacts",
        "metrics",
        "study_yaml_fingerprint",
        "model_revision",
        "sae_revision",
        "hook_configuration_hash",
        "layer_set_hash",
    }
    fields = {f.name for f in StageResult.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    missing = required - fields
    assert not missing, f"StageResult missing fields: {sorted(missing)}"

    result = dispatch_stage(
        "identity",
        study=_study(),
        freeze_status="unsealed",
        stack_loader=lambda _s: _FakeStack(),
    )
    assert result.ok is True
    assert isinstance(result.artifacts, dict)
    assert isinstance(result.metrics, dict)
    assert result.study_yaml_fingerprint
    assert len(result.study_yaml_fingerprint) == 64
    assert result.model_revision == _study().stack.model.revision
    assert result.layer_set_hash
    assert result.hook_configuration_hash
