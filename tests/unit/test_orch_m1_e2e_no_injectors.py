"""ORCH-033: unit e2e identity→full_study with fake stack_loader only."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch
import torch.nn as nn
import yaml

from epistemic_sycophancy.runner.cli import dispatch_stage
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

    def __call__(self, *, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 3, dtype=torch.float64)
        for i in range(batch):
            if i % 2 == 0:
                logits[i, -1, :] = torch.tensor([2.0, -1.0, 0.0])
            else:
                logits[i, -1, :] = torch.tensor([-1.0, 2.0, 0.0])
        return SimpleNamespace(logits=logits)


class _FakeStack:
    """Rich fake: identity residuals, LM scoring, FS projection, belief margins."""

    def __init__(self) -> None:
        self.model = _ToyCausalLM()
        self.tokenizer = _Tok()
        self.device = torch.device("cpu")
        decoder = torch.tensor(
            [[3.0, 0.0], [0.0, 4.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        self.saes = {
            17: SimpleNamespace(decoder_weight=decoder, layer=17),
        }

    def capture_layer_residuals(
        self, *, texts: list[str], layers: list[int]
    ) -> dict[int, torch.Tensor]:
        # Identical residuals → identity_passed.
        out = {}
        for layer in layers:
            out[layer] = torch.zeros(len(texts), 2, dtype=torch.float64)
        return out

    @contextmanager
    def install_hooks(self, **kwargs: Any) -> Iterator[None]:
        del kwargs
        yield

    def fs_projection_batch(self, **kwargs: Any) -> dict[str, Any]:
        qids = list(kwargs.get("question_ids") or ("q_fs_1", "q_fs_2"))
        n = len(qids)
        return {
            "layer": 17,
            "residual_gradients": torch.ones(n, 2, dtype=torch.float64),
            "latents": torch.tensor(
                [[1.0, 0.0, 0.0]] * n, dtype=torch.float64
            ),
            "question_ids": qids,
        }

    def score_belief_margins(
        self,
        *,
        belief_condition: str,
        question_ids,
        beta,
        order_regime: str = "CF",
    ):
        del beta, order_regime
        ids = list(question_ids)
        if belief_condition == "N":
            return {
                qid: (1.0 if i % 2 == 0 else -0.5) for i, qid in enumerate(ids)
            }
        if belief_condition == "IB":
            return {qid: (0.25,) for qid in ids}
        return {qid: (0.75,) for qid in ids}

    def margin_projection_batch(
        self,
        *,
        belief_condition: str,
        question_ids: tuple[str, ...],
        beta: tuple[float, ...],
    ):
        """GRAD-010: tiny linear-SAE batch; real coefficient_jacobian still runs."""
        del beta
        n = len(question_ids)
        decoder = self.saes[17].decoder_weight
        n_features = int(decoder.shape[0])
        # Match fs_projection_batch activity: feature 0 active.
        latents_row = torch.zeros(n_features, dtype=torch.float64)
        latents_row[0] = 1.0
        residual_g = torch.ones(int(decoder.shape[1]), dtype=torch.float64)
        scales = torch.linalg.vector_norm(decoder, dim=1)
        return {
            "layer": 17,
            "residual_gradients": residual_g.unsqueeze(0).expand(n, -1).clone(),
            "latents": latents_row.unsqueeze(0).expand(n, -1).clone(),
            "decoder": decoder,
            "feature_scales": scales,
            "question_ids": list(question_ids),
            "belief_condition": belief_condition,
        }


def _write_study_yaml(path: Path, artifact_dir: Path) -> None:
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
            "order_regimes": ["CF"],
            "feature_chunk_size": 2,
            "prompt_batch_size": 2,
            "smoke": {"question_ids": ["q_fs_1", "q_fs_2"]},
            "optimizer": {
                "kind": "projected_adam",
                "adam_lr": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1.0e-8,
                "adam_microbatch_questions": 1,
                "max_steps": 1,
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
def test_cli__fake_stack__identity_through_full_study_without_injector_kwargs(
    tmp_path: Path,
) -> None:
    """ORCH-033: only stack_loader injected; chain writes artifacts; no holdout."""
    from epistemic_sycophancy.config.load_study import load_study_config

    clear_stack_cache()
    cfg = tmp_path / "study.yaml"
    art = tmp_path / "art"
    _write_study_yaml(cfg, art)
    study = load_study_config(cfg)
    stack_loader = lambda _s: _FakeStack()
    corpus_kwargs = {
        "corpus_jsonl_paths": (FIXTURE_ROOT / "processed_mc0_tiny.jsonl",),
        "split_manifest_path": FIXTURE_ROOT / "split_manifest_tiny.csv",
    }

    stages = (
        "identity",
        "baseline_partitions",
        "feature_selection",
        "opt_smoke",
        "optimize",
        "freeze",
        "full_study",
    )
    for stage in stages:
        freeze = "sealed" if stage in {"full_study"} else "unsealed"
        result = dispatch_stage(
            stage,
            study=study,
            freeze_status=freeze,
            stack_loader=stack_loader,
            validation_question_ids=("q_val_1", "q_val_2"),
            optimization_question_ids=(
                ("q_fs_1", "q_fs_2") if stage != "feature_selection" else ()
            ),
            **corpus_kwargs,
        )
        assert result.ok, result.message

    assert (art / "identity" / "identity_result.json").is_file()
    assert (art / "baseline" / "partition_CF.json").is_file()
    assert (art / "feature_selection" / "common_pool.json").is_file()
    assert (art / "opt_smoke" / "opt_smoke_result.json").is_file()
    assert (art / "optimize" / "best_checkpoint.json").is_file()
    assert (art / "freeze" / "frozen_experiment_config.json").is_file()
    assert (art / "full_study" / "behavioral.json").is_file()
    assert not (art / "holdout").exists()
    frozen = json.loads(
        (art / "freeze" / "frozen_experiment_config.json").read_text(encoding="utf-8")
    )
    assert frozen.get("holdout_started") is False or "holdout_started" in frozen
