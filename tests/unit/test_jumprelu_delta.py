"""RUN-007: JumpReLU encode + Phase E latent update identity/suppression."""

from __future__ import annotations

import pytest
import torch

from epistemic_sycophancy.sae.jumprelu_delta import apply_additive_jumprelu_sae_delta


def _jumprelu(pre: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return pre * (pre > threshold).to(dtype=pre.dtype)


@pytest.mark.unit
def test_sae__jumprelu_latent_update__beta_zero_identity_and_suppression_nonincrease() -> None:
    """RUN-007: JumpReLU encode; β=0 identity; suppression does not increase latents."""
    torch.manual_seed(0)
    d_model = 4
    n_features = 5
    residual = torch.randn(d_model, dtype=torch.float64)
    encoder_weight = torch.randn(n_features, d_model, dtype=torch.float64)
    encoder_bias = torch.randn(n_features, dtype=torch.float64)
    threshold = torch.full((n_features,), 0.05, dtype=torch.float64)
    decoder_weight = torch.randn(n_features, d_model, dtype=torch.float64)

    # β=0 → exact residual identity (never reconstruction).
    identity = apply_additive_jumprelu_sae_delta(
        residual=residual,
        selected_indices=(1, 3),
        scales=(2.0, 0.5),
        beta=(0.0, 0.0),
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
    )
    assert torch.equal(identity, residual)

    # Reconstruction would differ from residual on this imperfect SAE.
    pre = residual @ encoder_weight.T + encoder_bias
    z = _jumprelu(pre, threshold)
    reconstruction = z @ decoder_weight
    assert not torch.allclose(reconstruction, residual)

    # Suppression: selected latents after update must be ≤ pre-update latents.
    selected = (0, 2, 4)
    scales = (1.0, 1.5, 0.25)
    beta = (-1.0, -0.5, -2.0)
    pre = residual @ encoder_weight.T + encoder_bias
    z = _jumprelu(pre, threshold)
    intervened = apply_additive_jumprelu_sae_delta(
        residual=residual,
        selected_indices=selected,
        scales=scales,
        beta=beta,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
    )
    # Recover z' from residual path: encode intervened? Better: recompute update algebra.
    alphas = torch.tensor(
        [s * b for s, b in zip(scales, beta)], dtype=torch.float64
    )
    z_prime = z.clone()
    for index, alpha in zip(selected, alphas):
        z_prime[index] = torch.relu(z[index] + alpha)
    for index in selected:
        assert float(z_prime[index]) <= float(z[index]) + 1e-12
        assert float(z_prime[index]) >= 0.0
    expected_delta = (z_prime - z) @ decoder_weight
    assert torch.allclose(intervened, residual + expected_delta, atol=1e-12, rtol=0.0)
