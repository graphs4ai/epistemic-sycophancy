"""ADAPT-012: belief scorer fails loudly when nonzero β cannot install hooks."""

from __future__ import annotations

import logging
from contextlib import contextmanager
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
from epistemic_sycophancy.logging.pipeline import PIPELINE_LOGGER_NAME
from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.runner.adapters.belief_scorer import build_belief_margin_scorer
from epistemic_sycophancy.sae.spec import SaeSiteSpec
from epistemic_sycophancy.stack.config import ExperimentStackConfig, HookSpec
from epistemic_sycophancy.stack.scoring import StackScoreBatch


class _ToyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")

    def __call__(self, *, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 3, dtype=torch.float64)
        logits[:, -1, 0] = 2.0
        logits[:, -1, 1] = -1.0
        return SimpleNamespace(logits=logits)


class _Tok:
    def __call__(self, texts, return_tensors="pt", padding=True):
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"A": [0], "B": [1]}[text]


class _Stack:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = _Tok()
        self.model = _ToyCausalLM()
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
        self.hook_calls.append(
            (tuple(selected_keys), tuple(scales), tuple(beta), tuple(prompt_lengths))
        )
        yield


def _study(*, feature_ids=(), feature_scales=(), coefficient_length=0) -> StudyConfig:
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
                layers=(17, 22),
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
            feature_ids=feature_ids,
            feature_scales=feature_scales,
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
            artifact_dir="/tmp/unused",
            order_regime="CF",
            feature_chunk_size=1024,
            prompt_batch_size=2,
            fs_coverage=StudyFsCoverageConfig(question_ids=("q0",)),
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
                max_steps=1,
            ),
        ),
    )


def _corpus() -> list[dict[str, object]]:
    from epistemic_sycophancy.runner.adapters.corpus import _normalize_processed_row

    raw = {
        "question_id": "q0",
        "split": "behavior_validation",
        "format": "mc0",
        "belief_condition": "neutral",
        "answer_order": "true-first",
        "correct_label": "A",
        "option_a": "Truth",
        "option_b": "False",
        "prompt": "Question: Q0?",
    }
    return [_normalize_processed_row(raw)]


@pytest.mark.unit
def test_belief_scorer__nonzero_beta_empty_features__raises() -> None:
    """ADAPT-012: nonzero β with empty feature_ids must fail loudly (DEC-102)."""
    study = _study(feature_ids=(), feature_scales=(), coefficient_length=0)
    stack = _Stack()
    scorer = build_belief_margin_scorer(
        study,
        stack,
        corpus=_corpus(),
        split_question_ids={"behavior_validation": ("q0",)},
        order_regime="CF",
    )
    with pytest.raises(ValueError, match="hooks|feature_ids|nonzero"):
        scorer(belief_condition="N", question_ids=("q0",), beta=(-0.5,))


@pytest.mark.unit
def test_belief_scorer__nonzero_beta__logs_hook_installation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ADAPT-012: log selected_feature_count / nonzero_beta_count / hooks / layers."""
    study = _study(
        feature_ids=((17, 1), (22, 3), (17, 5)),
        feature_scales=(1.0, 0.5, 2.0),
        coefficient_length=3,
    )
    stack = _Stack()
    scorer = build_belief_margin_scorer(
        study,
        stack,
        corpus=_corpus(),
        split_question_ids={"behavior_validation": ("q0",)},
        order_regime="CF",
    )

    def fake_score_batch(**kwargs):
        del kwargs
        return StackScoreBatch(
            margins=(0.5,),
            score_a=(1.0,),
            score_b=(0.5,),
            truthful_labels=("A",),
        )

    pipeline_logger = logging.getLogger(PIPELINE_LOGGER_NAME)
    pipeline_logger.addHandler(caplog.handler)
    try:
        with (
            caplog.at_level(logging.INFO, logger=PIPELINE_LOGGER_NAME),
            patch(
                "epistemic_sycophancy.runner.adapters.belief_scorer.score_batch_through_hooks",
                fake_score_batch,
            ),
        ):
            result = scorer(
                belief_condition="N",
                question_ids=("q0",),
                beta=(-0.5, 0.0, -1.0),
            )
    finally:
        pipeline_logger.removeHandler(caplog.handler)

    assert result == {"q0": 0.5}
    messages = [
        r.getMessage() for r in caplog.records if r.name == PIPELINE_LOGGER_NAME
    ]
    joined = "\n".join(messages)
    assert "selected_feature_count=3" in joined
    assert "nonzero_beta_count=2" in joined
    assert "hooks_enabled=true" in joined or "hooks_enabled=True" in joined
    assert "installed_layer_count=2" in joined
    del stack  # hooks entered inside score_batch; logging is the contract under test