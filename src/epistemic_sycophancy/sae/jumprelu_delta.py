"""JumpReLU encode + Phase E selected latent update (DEC-053 / RUN-007)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from epistemic_sycophancy.intervention.sae_delta import latent_delta_to_residual


def jumprelu(pre_activations: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Elementwise JumpReLU: pre * 1[pre > θ]."""
    return pre_activations * (pre_activations > threshold).to(
        dtype=pre_activations.dtype
    )


def apply_additive_jumprelu_sae_delta(
    *,
    residual: torch.Tensor,
    selected_indices: Sequence[int],
    scales: Sequence[float] | torch.Tensor,
    beta: Sequence[float] | torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    threshold: torch.Tensor,
    decoder_weight: torch.Tensor,
) -> torch.Tensor:
    """Return x' = x + (decode(z') - decode(z)); at β=0 return original x.

    Encode uses JumpReLU; selected update uses Phase E ReLU(z+α) (DEC-053).
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

    if not beta_tensor.requires_grad and bool(torch.all(beta_tensor == 0)):
        return residual

    pre = residual @ encoder_weight.T + encoder_bias
    latents = jumprelu(pre, threshold.to(dtype=residual.dtype, device=residual.device))
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
