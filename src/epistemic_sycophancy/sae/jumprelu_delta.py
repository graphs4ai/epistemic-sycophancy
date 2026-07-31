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
    """Return x' = x + (decode(z') - decode(z)).

    Encode uses JumpReLU; selected update uses Phase E ReLU(z+α) (DEC-053).
    At β=0, α=0 ⇒ z'=z ⇒ Δx=0 ⇒ x'=x (DEC-086; never SAE reconstruction).

    Latent algebra runs in float32 when ``residual`` is lower precision so that
    small α on large latents is not lost to bf16 ULP (WIRE-002).
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

    work_dtype = (
        torch.float32
        if residual.dtype in (torch.bfloat16, torch.float16)
        else residual.dtype
    )
    residual_w = residual.to(dtype=work_dtype)
    encoder_weight_w = encoder_weight.to(dtype=work_dtype, device=residual.device)
    encoder_bias_w = encoder_bias.to(dtype=work_dtype, device=residual.device)
    threshold_w = threshold.to(dtype=work_dtype, device=residual.device)
    decoder_weight_w = decoder_weight.to(dtype=work_dtype, device=residual.device)
    beta_w = beta_tensor.to(dtype=work_dtype, device=residual.device)
    scales_w = scales_tensor.to(dtype=work_dtype, device=residual.device)

    pre = residual_w @ encoder_weight_w.T + encoder_bias_w
    latents = jumprelu(pre, threshold_w)
    alphas = scales_w * beta_w
    alpha_full = torch.zeros_like(latents)
    index = torch.as_tensor(
        list(selected_indices), device=residual.device, dtype=torch.long
    )
    alpha_full = alpha_full.scatter(0, index, alphas)
    latents_prime = torch.relu(latents + alpha_full)
    delta = latent_delta_to_residual(
        latents_prime=latents_prime,
        latents=latents,
        decoder_weight=decoder_weight_w,
    )
    return (residual_w + delta).to(dtype=residual.dtype)
