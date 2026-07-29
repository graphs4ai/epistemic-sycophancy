"""REAL-007: peak CUDA memory regression."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.evaluation.real_model_smoke import (
    real_model_peak_cuda_memory,
)
from tests.real_model._pin import MODEL_ID, MODEL_REVISION


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__memory_regression__peak_cuda_within_budget() -> None:
    """REAL-007: peak CUDA memory for baseline/hooked/Adam stays within DEC-045 budget."""
    if not torch.cuda.is_available():
        pytest.fail(
            "CUDA unavailable; REAL-007 must be recorded blocked (DEC-045), "
            "never faked on CPU"
        )
    report = real_model_peak_cuda_memory(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompt="Answer:",
        selected_indices=(0, 2, 5),
        scales=(1.5, 1.5, 1.5),
        seed=0,
    )
    budget = max(512 * 1024 * 1024, 4 * report.baseline_peak_bytes)
    assert report.baseline_peak_bytes <= budget
    assert report.hooked_peak_bytes <= budget
    assert report.adam_backward_peak_bytes <= budget
