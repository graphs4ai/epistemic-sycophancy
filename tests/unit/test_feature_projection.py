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
spec_gradient = _toy_gradients.spec_gradient


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


@pytest.mark.unit
def test_feature_projection__toy_vectors__match_decoder_dot_products() -> None:
    """FEAT-003: g=[2,-1] against rows [1,0], [0,3], [1,1] gives h=[2,-3,1].

    Hand-derived from spec §11: <g,f0> = 2, <g,f1> = -3, <g,f2> = 1.
    """
    projection = project_residual_gradient(
        gradient=spec_gradient(), decoder=spec_decoder()
    )

    assert projection.tolist() == [2.0, -3.0, 1.0]


@pytest.mark.unit
def test_feature_projection__feature_chunking__matches_dense_matrix_multiplication() -> None:
    """FEAT-018: chunked projection equals dense matmul, including uneven final chunk."""
    torch.manual_seed(0)
    batch, d_model, n_features = 5, 4, 11
    gradient = torch.randn(batch, d_model, dtype=torch.float64)
    decoder = torch.randn(n_features, d_model, dtype=torch.float64)

    dense = project_residual_gradient(gradient=gradient, decoder=decoder)
    # Chunk size 3 → uneven final chunk of 2 (11 = 3+3+3+2).
    chunked = project_residual_gradient(
        gradient=gradient, decoder=decoder, feature_chunk_size=3
    )
    assert torch.allclose(chunked, dense, atol=1e-10, rtol=1e-9)

    # Multiple "layers": two independent decoder matrices must not mix.
    decoder_l0 = torch.randn(7, d_model, dtype=torch.float64)
    decoder_l1 = torch.randn(9, d_model, dtype=torch.float64)
    for decoder_layer, chunk in ((decoder_l0, 4), (decoder_l1, 5)):
        dense_layer = project_residual_gradient(
            gradient=gradient, decoder=decoder_layer
        )
        chunked_layer = project_residual_gradient(
            gradient=gradient, decoder=decoder_layer, feature_chunk_size=chunk
        )
        assert torch.allclose(chunked_layer, dense_layer, atol=1e-10, rtol=1e-9)
        assert chunked_layer.shape == (batch, decoder_layer.shape[0])
