"""Raw decoder-direction projection of residual gradients (Phase F)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.feature_selection import project_residual_gradient

_TOY_GRADIENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "feature_selection"
    / "toy_gradients.py"
)
_spec = importlib.util.spec_from_file_location(
    "toy_gradients_projection", _TOY_GRADIENTS_PATH
)
assert _spec is not None and _spec.loader is not None
_toy_gradients = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy_gradients)
spec_decoder = _toy_gradients.spec_decoder


@pytest.mark.unit
def test_feature_projection__gradient_times_decoder_transpose__returns_feature_dimension() -> (
    None
):
    """FEAT-002: [batch, d_model] x [n_features, d_model]^T -> [batch, n_features]."""
    decoder = spec_decoder()
    gradient = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, -1.0],
            [-3.0, 0.5],
        ],
        dtype=torch.float64,
    )

    projection = project_residual_gradient(gradient=gradient, decoder=decoder)

    assert projection.shape == (gradient.shape[0], decoder.shape[0])
    assert projection.dtype == torch.float64
