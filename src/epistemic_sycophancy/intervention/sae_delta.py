"""Additive SAE latent updates and residual deltas."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def normalized_coefficients(
    *,
    scales: Sequence[float],
    beta: Sequence[float],
) -> list[float]:
    """Return α_j = s_j β_j for each selected feature.

    Shapes: scales and beta must have equal length m (selected features).
    """
    return [float(s) * float(b) for s, b in zip(scales, beta)]


def apply_selected_latent_update(
    *,
    latents: Sequence[float],
    selected_indices: Sequence[int],
    scales: Sequence[float],
    beta: Sequence[float],
) -> list[float]:
    """Apply α = s⊙β to selected indices only (DEC-018).

    Non-selected latents remain unchanged. Full width is len(latents).
    ReLU: z'_j = max(0, z_j + α_j) on selected indices.
    """
    alphas = normalized_coefficients(scales=scales, beta=beta)
    updated = [float(z) for z in latents]
    for index, alpha in zip(selected_indices, alphas):
        updated[index] = max(0.0, updated[index] + alpha)
    return updated


def latent_delta_to_residual(
    *,
    latents_prime: torch.Tensor,
    latents: torch.Tensor,
    decoder_weight: torch.Tensor,
) -> torch.Tensor:
    """Return (z' - z) W_dec for a linear decoder (DEC-016).

    Shapes:
      latents_prime, latents: [..., n_features]
      decoder_weight: [n_features, d_model]
      returns: [..., d_model]
    """
    return (latents_prime - latents) @ decoder_weight


def apply_additive_sae_delta(
    *,
    residual: torch.Tensor,
    selected_indices: Sequence[int],
    scales: Sequence[float] | torch.Tensor,
    beta: Sequence[float] | torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    decoder_weight: torch.Tensor,
) -> torch.Tensor:
    """Return x' = x + (decode(z') - decode(z)).

    At β=0, α=0 ⇒ z'=z ⇒ Δx=0 ⇒ x'=x (DEC-086), never the SAE reconstruction
    decode(encode(x)). The additive path always stays in the autograd graph
    when ``beta`` requires grad (SAE-013).
    """
    beta_tensor = (
        beta
        if isinstance(beta, torch.Tensor)
        else torch.tensor(list(beta), dtype=residual.dtype, device=residual.device)
    )
    scales_tensor = (
        scales
        if isinstance(scales, torch.Tensor)
        else torch.tensor(list(scales), dtype=residual.dtype, device=residual.device)
    )

    latents = torch.relu(residual @ encoder_weight.T + encoder_bias)
    alphas = scales_tensor * beta_tensor
    alpha_full = torch.zeros_like(latents)
    index = torch.as_tensor(
        list(selected_indices), device=residual.device, dtype=torch.long
    )
    alpha_full = alpha_full.scatter(0, index, alphas)
    latents_prime = torch.relu(latents + alpha_full)
    delta = latent_delta_to_residual(
        latents_prime=latents_prime,
        latents=latents,
        decoder_weight=decoder_weight,
    )
    return residual + delta
