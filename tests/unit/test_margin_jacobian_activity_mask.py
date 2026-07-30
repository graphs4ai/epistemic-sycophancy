"""GRAD-005: JumpReLU / activity mask on margin Jacobian path."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_margin_jacobian__inactive_latent__zero_selected_derivative() -> None:
    """GRAD-005: 1[z>0] on post-encode latents; inactive selected feature → 0.

    Aligns DEC-053 / DEC-084 with FEAT-007 spirit on the margin ∂M/∂β path.
    """
    from epistemic_sycophancy.runner.adapters.margin_jacobian import (
        project_selected_margin_jacobian,
    )

    decoder = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    residual_gradient = torch.tensor([2.0, -1.0], dtype=torch.float64)
    # Feature 0 inactive (z=0); feature 1 active.
    latents = torch.tensor([0.0, 1.5], dtype=torch.float64)
    feature_scales = torch.tensor([3.0, 2.0], dtype=torch.float64)

    jac_row = project_selected_margin_jacobian(
        residual_gradient=residual_gradient,
        latents=latents,
        decoder=decoder,
        feature_scales=feature_scales,
        selected_indices=(0, 1),
    )
    # h = g @ W_dec^T = [2, -1]; J = s * 1[z>0] * h = [0, 2*(-1)] = [0, -2]
    assert jac_row.tolist() == [0.0, -2.0]
