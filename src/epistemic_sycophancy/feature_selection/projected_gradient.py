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


def coefficient_jacobian(
    *,
    raw_projection: torch.Tensor,  # [..., n_features]
    latents: torch.Tensor,  # [..., n_features]
    feature_scales: torch.Tensor,  # [n_features]
) -> torch.Tensor:  # [..., n_features]
    """Return the exact local derivative w.r.t. normalized coefficients.

    J_j = s_j * 1[z_j > 0] * h_j (AGENTS.md §5.8). The activity mask is
    strict: a latent sitting exactly at zero contributes nothing under a
    feasible nonpositive coefficient change.
    """
    activity_mask = (latents > 0).to(raw_projection.dtype)
    return feature_scales * activity_mask * raw_projection
