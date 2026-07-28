"""Additive SAE latent updates and residual deltas."""

from __future__ import annotations

from collections.abc import Sequence


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
    """
    alphas = normalized_coefficients(scales=scales, beta=beta)
    updated = [float(z) for z in latents]
    for index, alpha in zip(selected_indices, alphas):
        updated[index] = updated[index] + alpha
    return updated
