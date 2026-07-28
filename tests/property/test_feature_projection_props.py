"""Property tests for raw decoder projection (Phase F)."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.feature_selection import project_residual_gradient


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=8),
    d_model=st.integers(min_value=1, max_value=6),
    n_features=st.integers(min_value=1, max_value=8),
    data=st.data(),
)
def test_feature_projection__mean_gradient_then_project__equals_project_then_mean_without_masks(
    batch: int,
    d_model: int,
    n_features: int,
    data: st.DataObject,
) -> None:
    """FEAT-015: mean(g) W^T = mean(g W^T) when there are no masks or scales."""
    elements = st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    )
    gradient = torch.tensor(
        data.draw(
            st.lists(
                st.lists(elements, min_size=d_model, max_size=d_model),
                min_size=batch,
                max_size=batch,
            )
        ),
        dtype=torch.float64,
    )
    decoder = torch.tensor(
        data.draw(
            st.lists(
                st.lists(elements, min_size=d_model, max_size=d_model),
                min_size=n_features,
                max_size=n_features,
            )
        ),
        dtype=torch.float64,
    )

    mean_then_project = project_residual_gradient(
        gradient=gradient.mean(dim=0), decoder=decoder
    )
    project_then_mean = project_residual_gradient(
        gradient=gradient, decoder=decoder
    ).mean(dim=0)

    assert torch.allclose(mean_then_project, project_then_mean, atol=1e-10, rtol=1e-9)
