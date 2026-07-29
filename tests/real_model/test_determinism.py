"""REAL-004: real-model deterministic repeated scoring."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.evaluation.real_model_smoke import score_real_model_batch
from tests.real_model._pin import ATOL, MODEL_ID, MODEL_REVISION, RTOL

PROMPTS = ("Answer:", "Q: sky? Answer:")
SELECTED = (0, 2, 5)
SCALES = (1.5, 1.5, 1.5)
BETA = (-0.1, 0.0, -0.05)


@pytest.mark.real_model
@pytest.mark.slow
def test_real_model__fixed_seeds__repeated_scoring_is_deterministic() -> None:
    """REAL-004: repeated scoring with fixed seeds agrees."""
    first = score_real_model_batch(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompts=PROMPTS,
        beta=BETA,
        selected_indices=SELECTED,
        scales=SCALES,
        seed=0,
    )
    second = score_real_model_batch(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompts=PROMPTS,
        beta=BETA,
        selected_indices=SELECTED,
        scales=SCALES,
        seed=0,
    )
    assert torch.equal(first.logits, second.logits) or torch.allclose(
        first.logits, second.logits, atol=ATOL, rtol=RTOL
    )
    assert first.margins == second.margins
    assert first.labels == second.labels
