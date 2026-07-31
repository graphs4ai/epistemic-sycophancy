"""ORCH-021: build_score_fn from StudyConfig + InterventionStack (β=0 neutrals)."""

from __future__ import annotations

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
    StudySmokeConfig,
)
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.prompts.render import render_mc0_subset
from epistemic_sycophancy.runner.adapters.corpus import load_processed_mc0_corpus
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.scoring.margins import truthful_margin
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.scoring import score_batch_with_lm_logits

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "adapters"
PROCESSED_JSONL = FIXTURE_ROOT / "processed_mc0_tiny.jsonl"


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
        logits[:, -1, :] = self.vocab_logits
        return SimpleNamespace(logits=logits)


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


class _FakeStack:
    def __init__(self, model: _ToyCausalLM, tokenizer: _Tok) -> None:
        self.model = model
        self.tokenizer = tokenizer
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
            artifact_dir=artifact_dir,
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=1,
            smoke=StudySmokeConfig(question_ids=("q_fs_1", "q_fs_2")),
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


@pytest.mark.unit
def test_adapters__build_score_fn__beta0_neutral_margins_match_hand_toy_stack(
    tmp_path: Path,
) -> None:
    """ORCH-021: build_score_fn scores neutrals at β=0 via score_batch_through_hooks."""
    from epistemic_sycophancy.runner.adapters.score import build_score_fn

    vocab_logits = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float64)
    stack = _FakeStack(_ToyCausalLM(vocab_logits), _Tok())
    study = _study(artifact_dir=str(tmp_path / "art"))
    corpus = load_processed_mc0_corpus(jsonl_paths=(PROCESSED_JSONL,), ro_seed=42)
    split_ids = {
        "feature_selection": ("q_fs_1", "q_fs_2"),
        "optimization": ("q_opt_1",),
    }

    score_fn = build_score_fn(
        study,
        stack,
        corpus=corpus,
        split_question_ids=split_ids,
        order_regime="CF",
        belief_condition="N",
    )
    margins = score_fn(("q_fs_1", "q_fs_2"))
    assert set(margins) == {"q_fs_1", "q_fs_2"}

    # Independent hand path: render + unhooked LM scoring (β=0 ≡ no hooks).
    rendered = render_mc0_subset(
        corpus_rows=corpus,
        smoke=study.run.smoke,
        split_question_ids=split_ids,
        order_regime="CF",
        belief_condition="N",
    )
    expected_batch = score_batch_with_lm_logits(
        model=stack.model,
        tokenizer=stack.tokenizer,
        prompts=[r.text for r in rendered],
        continuation_token_ids_A=[0],
        continuation_token_ids_B=[1],
        truthful_labels=tuple(r.truthful_label for r in rendered),
        device=stack.device,
    )
    for row, margin in zip(rendered, expected_batch.margins, strict=True):
        assert margins[row.question_id] == pytest.approx(margin)
    # Semantic margin: M = s_truth - s_incorrect via library truthful_margin.
    assert expected_batch.margins[0] == truthful_margin(
        score_a=expected_batch.score_a[0],
        score_b=expected_batch.score_b[0],
        truthful_label=rendered[0].truthful_label,
    )
