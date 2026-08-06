"""ORCH-032: CLI run_cli production path needs no injector kwargs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from epistemic_sycophancy.runner.cli import run_cli
from epistemic_sycophancy.runner.identity import clear_stack_cache

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


@pytest.mark.unit
def test_cli__run_cli__config_only_no_injector_kwargs_for_baseline(
    tmp_path: Path,
) -> None:
    """ORCH-032: run_cli(--config) baseline succeeds without score_fn (DEC-065)."""
    clear_stack_cache()
    cfg = tmp_path / "study.yaml"
    art = tmp_path / "art"
    _write_tiny_study_yaml(cfg, art)
    code = run_cli(
        ["baseline_partitions", "--config", str(cfg)],
        stack_loader=lambda _study: _FakeStack(),
        corpus_jsonl_paths=(FIXTURE_ROOT / "processed_mc0_tiny.jsonl",),
        split_manifest_path=FIXTURE_ROOT / "split_manifest_tiny.csv",
    )
    assert code == 0
    assert (art / "baseline" / "partition_CF.json").is_file()
