"""ORCH-001: dispatch identity runs real β=0 identity (not fingerprint-only)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

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
            artifact_dir="artifacts/smokes/layer17_n2",
            order_regimes=("CF", "IF", "RO"),
            feature_chunk_size=1024,
            prompt_batch_size=1,
            smoke=StudySmokeConfig(
                n_questions=2, split="feature_selection", seed=0
            ),
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
                n_questions=4,
            ),
        ),
    )


class _FakeIdentityStack:
    """Toy stack: β=0 hooks leave residuals unchanged (ORCH-001 unit path)."""

    def __init__(self) -> None:
        self.config = _study().stack
        self._residual = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
        self.capture_calls = 0
        self.hook_installs = 0

    def capture_layer_residuals(
        self,
        *,
        texts: Sequence[str],
        layers: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        del texts
        self.capture_calls += 1
        return {int(layer): self._residual.clone() for layer in layers}

    @contextmanager
    def install_hooks(
        self,
        *,
        selected_keys: Sequence[tuple[int, int]],
        scales: Sequence[float],
        beta: Sequence[float],
        prompt_lengths: Sequence[int],
    ) -> Iterator[None]:
        del selected_keys, scales, prompt_lengths
        self.hook_installs += 1
        if any(float(b) != 0.0 for b in beta):
            raise AssertionError("identity stage must use β=0")
        yield


class _FakeBrokenIdentityStack(_FakeIdentityStack):
    """Hooked residuals differ from unhooked even at β=0 (identity failure)."""

    def capture_layer_residuals(
        self,
        *,
        texts: Sequence[str],
        layers: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        del texts
        self.capture_calls += 1
        # Alternate unhooked vs hooked by call parity after hooks installed.
        if self.hook_installs > 0:
            return {
                int(layer): self._residual.clone() + 1.0 for layer in layers
            }
        return {int(layer): self._residual.clone() for layer in layers}


@pytest.mark.unit
def test_dispatch__identity_failure__sets_ok_false_and_blocks_require_identity_gate() -> None:
    """ORCH-002: failed identity → StageResult.ok=False; require_identity_gate blocks."""
    from epistemic_sycophancy.reproducibility.phase_gates import (
        OptimizationBlockedError,
        require_identity_gate,
    )
    from epistemic_sycophancy.runner.cli import dispatch_stage
    from epistemic_sycophancy.runner.identity import clear_stack_cache

    clear_stack_cache()
    stack = _FakeBrokenIdentityStack()

    result = dispatch_stage(
        "identity",
        study=_study(),
        freeze_status="unsealed",
        stack_loader=lambda _study: stack,
    )

    assert result.ok is False
    assert result.metrics.get("identity_passed") is False
    assert result.metrics.get("max_abs_diff", 0.0) > 0.0

    with pytest.raises(OptimizationBlockedError):
        require_identity_gate(identity_passed=bool(result.metrics["identity_passed"]))


@pytest.mark.unit
def test_dispatch__identity__beta_zero_identity_on_smoke_prompts_returns_structured_stage_result() -> None:
    """ORCH-001: identity loads stack, checks β=0 residual identity, structured StageResult."""
    from epistemic_sycophancy.runner.cli import dispatch_stage

    stack = _FakeIdentityStack()
    loads: list[Any] = []

    def stack_loader(study: StudyConfig) -> Any:
        loads.append(study)
        return stack

    result = dispatch_stage(
        "identity",
        study=_study(),
        freeze_status="unsealed",
        stack_loader=stack_loader,
    )

    assert result.ok is True
    assert result.stage == "identity"
    assert loads, "identity must call stack_loader (DEC-065)"
    assert stack.capture_calls >= 2, "must compare unhooked vs hooked residuals"
    assert stack.hook_installs >= 1
    assert "fingerprint-only" not in result.message
    assert "study_fp=" not in result.message or "identity_passed" in result.message
    # Structured outcome (not fingerprint-only stub)
    assert getattr(result, "metrics", None) is not None
    assert result.metrics.get("identity_passed") is True
    assert result.metrics.get("max_abs_diff", 1.0) == pytest.approx(0.0)


@pytest.mark.unit
def test_resolve_stack__injected_loader__does_not_poison_default_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEC-083: injecting stack_loader must not poison the process cache for later default loads."""
    from epistemic_sycophancy.runner.identity import clear_stack_cache, resolve_stack

    clear_stack_cache()
    injected = SimpleNamespace(kind="injected")
    default_sentinel = SimpleNamespace(kind="default")
    monkeypatch.setattr(
        "epistemic_sycophancy.runner.identity._default_stack_loader",
        lambda _study: default_sentinel,
    )

    study = _study()
    assert resolve_stack(study, stack_loader=lambda _s: injected) is injected
    # A later default resolve (no loader) must load via default, not reuse the inject.
    assert resolve_stack(study) is default_sentinel

