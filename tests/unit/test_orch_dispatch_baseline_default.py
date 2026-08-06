"""ORCH-027: baseline builds score_fn when None (DEC-075/077)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q_fs_1", "q_fs_2")),
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
                n_questions=2,
            ),
        ),
    )


@pytest.mark.unit
def test_dispatch__baseline_partitions__builds_score_fn_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORCH-027: score_fn=None → build from stack; single study order_regime."""
    monkeypatch.chdir(tmp_path)
    # Point corpus bridge at fixtures without depending on repo cwd.
    art = tmp_path / "art"
    study = _study(artifact_dir=str(art))

    result = dispatch_stage(
        "baseline_partitions",
        study=study,
        freeze_status="unsealed",
        stack_loader=lambda _s: _FakeStack(),
        # corpus inject via kwargs if supported; else default fixture path override
        corpus_jsonl_paths=(FIXTURE_ROOT / "processed_mc0_tiny.jsonl",),
        split_manifest_path=FIXTURE_ROOT / "split_manifest_tiny.csv",
        score_fn=None,
    )
    assert result.ok
    path = art / "baseline" / "partition_CF.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["order_regime"] == "CF"
    assert "q_plus" in payload
    assert not (art / "baseline" / "partition_IF.json").exists()
