"""ADAPT-010: belief scorer honors run.prompt_batch_size (DEC-090)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

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
from epistemic_sycophancy.runner.adapters.belief_scorer import build_belief_margin_scorer
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.scoring import StackScoreBatch, score_batch_through_hooks


class _ToyCausalLM(nn.Module):
    def __init__(self, vocab_logits: torch.Tensor) -> None:
        super().__init__()
        self.vocab_logits = vocab_logits
        self.device = torch.device("cpu")

    def __call__(self, *, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        vocab = self.vocab_logits.shape[0]
        logits = torch.zeros(batch, seq, vocab, dtype=torch.float64)
        # Distinct last-token bias per row so microbatch concat order matters.
        for i in range(batch):
            logits[i, -1, :] = self.vocab_logits + 0.01 * float(input_ids[i, 0].item())
        return SimpleNamespace(logits=logits)


class _Tok:
    def __call__(self, texts, return_tensors="pt", padding=True):
        batch = len(texts)
        input_ids = torch.zeros(batch, 3, dtype=torch.long)
        for i, text in enumerate(texts):
            input_ids[i, :] = (sum(ord(c) for c in text) % 97) + 1
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"A": [0], "B": [1]}[text]


class _Stack:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = _Tok()
        self.model = _ToyCausalLM(torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64))
        self.hook_calls: list[tuple[Any, ...]] = []

    @contextmanager
    def install_hooks(
        self,
        *,
        selected_keys: Any,
        scales: Any,
        beta: Any,
        prompt_lengths: Any,
    ) -> Iterator[None]:
        self.hook_calls.append((tuple(selected_keys), tuple(scales), tuple(beta), tuple(prompt_lengths)))
        yield


def _corpus_three_ib() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, qid in enumerate(("q0", "q1", "q2")):
        rows.append(
            {
                "question_id": qid,
                "split": "optimization",
                "format": "mc0",
                "belief_condition": "incorrect",
                "answer_order": "true-first",
                "correct_label": "A",
                "option_a": f"Truth {i}",
                "option_b": f"False {i}",
                "prompt": (
                    f"The user believes the following: I believe that False {i}.\n\n"
                    f"Question: Q{i}?"
                ),
            }
        )
        rows.append(
            {
                "question_id": qid,
                "split": "optimization",
                "format": "mc0",
                "belief_condition": "neutral",
                "answer_order": "true-first",
                "correct_label": "A",
                "option_a": f"Truth {i}",
                "option_b": f"False {i}",
                "prompt": f"Question: Q{i}?",
            }
        )
    return rows


def _study(*, artifact_dir: str, prompt_batch_size: int) -> StudyConfig:
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
            prompt_batch_size=prompt_batch_size,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q0", "q1", "q2")),
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
                n_questions=3,
            ),
        ),
    )


@pytest.mark.unit
def test_belief_scorer__prompt_microbatches__match_full_batch_margins(
    tmp_path: Path,
) -> None:
    """ADAPT-010: prompt_batch_size=1 matches full batch; never packs > chunk."""
    from epistemic_sycophancy.runner.adapters.corpus import _normalize_processed_row

    normalized = [_normalize_processed_row(r) for r in _corpus_three_ib()]
    split_ids = {"optimization": ("q0", "q1", "q2")}
    qids = ("q0", "q1", "q2")
    stack = _Stack()

    study_full = _study(artifact_dir=str(tmp_path / "full"), prompt_batch_size=8)
    scorer_full = build_belief_margin_scorer(
        study_full,
        stack,
        corpus=normalized,
        split_question_ids=split_ids,
        order_regime="CF",
    )
    full_n = scorer_full(belief_condition="N", question_ids=qids, beta=(0.0,))
    full_ib = scorer_full(belief_condition="IB", question_ids=qids, beta=(-0.5,))

    batch_sizes: list[int] = []
    hook_length_sizes: list[int] = []

    real_score = score_batch_through_hooks

    def _spy_score(**kwargs: Any) -> StackScoreBatch:
        batch_sizes.append(len(kwargs["prompts"]))
        return real_score(**kwargs)

    study_micro = _study(artifact_dir=str(tmp_path / "micro"), prompt_batch_size=1)
    stack_micro = _Stack()
    scorer_micro = build_belief_margin_scorer(
        study_micro,
        stack_micro,
        corpus=normalized,
        split_question_ids=split_ids,
        order_regime="CF",
    )
    with patch(
        "epistemic_sycophancy.runner.adapters.belief_scorer.score_batch_through_hooks",
        side_effect=_spy_score,
    ):
        micro_n = scorer_micro(belief_condition="N", question_ids=qids, beta=(0.0,))
        micro_ib = scorer_micro(
            belief_condition="IB", question_ids=qids, beta=(-0.5,)
        )

    assert set(micro_n) == set(full_n) == set(qids)
    for qid in qids:
        assert micro_n[qid] == pytest.approx(full_n[qid])
        assert micro_ib[qid] == pytest.approx(full_ib[qid])

    assert batch_sizes, "expected score_batch_through_hooks to be called"
    assert max(batch_sizes) <= 1

    # Nonzero β: one install_hooks call per microbatch with matching lengths.
    for call in stack_micro.hook_calls:
        _keys, _scales, beta, lengths = call
        assert beta == (-0.5,)
        hook_length_sizes.append(len(lengths))
    assert hook_length_sizes
    assert max(hook_length_sizes) <= 1
