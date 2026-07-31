"""REAL-005: β-only backward viability on pinned real model."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.evaluation.real_model_checks import (
    real_model_beta_backward_viability,
)
from tests.real_model._pin import MODEL_ID, MODEL_REVISION


@pytest.mark.real_model
@pytest.mark.slow
def test_real_model__backward_to_beta__model_and_sae_parameters_have_no_grad() -> None:
    """REAL-005: one backward to β; model/SAE grads absent; β grad finite."""
    report = real_model_beta_backward_viability(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompt="Answer:",
        selected_indices=(0, 2, 5),
        scales=(1.5, 1.5, 1.5),
        seed=0,
    )
    assert report.beta_grad_finite is True
    assert report.model_params_require_grad is False
    assert report.sae_params_require_grad is False
    assert report.model_grads_all_none is True
    assert report.sae_grads_all_none is True
