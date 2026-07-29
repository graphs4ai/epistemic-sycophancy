"""REAL-003: real-model β=0 identity."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.evaluation.real_model_smoke import score_real_model_batch
from tests.real_model._pin import ATOL, MODEL_ID, MODEL_REVISION, RTOL

PROMPTS = ("Answer:", "Q: sky? Answer:")
SELECTED = (0, 2, 5)
SCALES = (1.0, 1.0, 1.0)


@pytest.mark.real_model
@pytest.mark.slow
def test_real_model__zero_beta__logits_margins_and_labels_match_unhooked() -> None:
    """REAL-003: β=0 hooked scores match unhooked within real-model tolerances."""
    unhooked = score_real_model_batch(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompts=PROMPTS,
        beta=(0.0, 0.0, 0.0),
        selected_indices=SELECTED,
        scales=SCALES,
    )
    hooked = score_real_model_batch(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompts=PROMPTS,
        beta=(0.0, 0.0, 0.0),
        selected_indices=SELECTED,
        scales=SCALES,
    )
    assert torch.allclose(hooked.logits, unhooked.logits, atol=ATOL, rtol=RTOL)
    assert hooked.margins == pytest.approx(unhooked.margins, abs=ATOL, rel=RTOL)
    assert hooked.labels == unhooked.labels
