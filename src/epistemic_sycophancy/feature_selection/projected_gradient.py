"""Decoder-direction projection of residual gradients (Phase F)."""

from __future__ import annotations

import torch


def project_residual_gradient(
    *,
    gradient: torch.Tensor,  # [..., d_model]
    decoder: torch.Tensor,  # [n_features, d_model]
) -> torch.Tensor:  # [..., n_features]
    """Return the raw projection h = g W_dec^T onto decoder directions."""
    return gradient @ decoder.T
