"""REAL-002: real-model residual hook tensor contract."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.intervention.hooks import inspect_real_model_hook_contract
from tests.real_model._pin import MODEL_ID, MODEL_REVISION


@pytest.mark.real_model
@pytest.mark.slow
def test_real_model__hook_contract__shape_dtype_device_indexing_and_decoder_width() -> None:
    """REAL-002: hook module, shape, dtype, device, indexing, decoder width."""
    report = inspect_real_model_hook_contract(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        prompts=("Answer:", "Q: sky? Answer:"),
        n_features=8,
        dtype=torch.float32,
    )
    assert report.hook_module_name.endswith("transformer.h") or "transformer" in report.hook_module_name
    assert report.activation_shape[-1] == report.d_model
    assert report.activation_dtype == torch.float32
    assert str(report.device).startswith("cpu") or "cuda" in str(report.device)
    assert report.sequence_index_policy == "last_non_pad"
    assert report.decoder_width == report.d_model
    assert report.compatible is True
