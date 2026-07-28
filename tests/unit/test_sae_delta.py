"""SAE intervention delta tests (Phase E)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from epistemic_sycophancy.intervention.sae_delta import (
    apply_selected_latent_update,
    latent_delta_to_residual,
    normalized_coefficients,
)

_TOY_SAE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "intervention" / "toy_sae.py"
)
_spec = importlib.util.spec_from_file_location("toy_sae", _TOY_SAE_PATH)
assert _spec is not None and _spec.loader is not None
_toy_sae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy_sae)
decode = _toy_sae.decode
decoder_weight = _toy_sae.decoder_weight


@pytest.mark.unit
def test_intervention__normalized_beta__is_scaled_featurewise() -> None:
    """SAE-001: α_j = s_j β_j featurewise."""
    scales = [2.0, 0.5]
    beta = [-1.0, -2.0]
    alpha = normalized_coefficients(scales=scales, beta=beta)
    assert alpha == pytest.approx([-2.0, -1.0], abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_intervention__selected_features__leave_all_other_latents_unchanged() -> None:
    """SAE-002: only selected latents change; others remain identical (DEC-018)."""
    latents = [1.0, 2.0, 3.0, 4.0]
    selected_indices = [0, 2]
    # α on selected only: α_0 = -0.5, α_2 = -1.0 (via scales⊙β)
    scales = [1.0, 1.0]
    beta = [-0.5, -1.0]
    updated = apply_selected_latent_update(
        latents=latents,
        selected_indices=selected_indices,
        scales=scales,
        beta=beta,
    )
    assert updated[1] == latents[1]
    assert updated[3] == latents[3]
    assert updated[0] != latents[0]
    assert updated[2] != latents[2]


@pytest.mark.unit
def test_intervention__suppression_crossing_zero__clamps_at_zero() -> None:
    """SAE-003: z'_j = ReLU(z_j + α_j); suppression past zero clamps at 0."""
    latents = [0.4]
    selected_indices = [0]
    scales = [1.0]
    beta = [-1.0]  # α = -1.0
    updated = apply_selected_latent_update(
        latents=latents,
        selected_indices=selected_indices,
        scales=scales,
        beta=beta,
    )
    assert updated[0] == pytest.approx(0.0, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_intervention__linear_decoder__delta_decode_equals_latent_delta_times_decoder() -> None:
    """SAE-006: decode(z') - decode(z) = (z' - z) W_dec (DEC-016, bf16)."""
    dtype = torch.bfloat16
    w_dec = decoder_weight(dtype=dtype)
    latents = torch.tensor([1.0, 0.5, 0.25], dtype=dtype)
    latents_prime = torch.tensor([0.5, 0.5, 0.0], dtype=dtype)

    delta_via_decode = decode(latents_prime, decoder_weight=w_dec) - decode(
        latents, decoder_weight=w_dec
    )
    delta_via_matmul = latent_delta_to_residual(
        latents_prime=latents_prime,
        latents=latents,
        decoder_weight=w_dec,
    )
    assert torch.allclose(
        delta_via_matmul,
        delta_via_decode,
        atol=5e-3,
        rtol=1e-4,
    )
    # Hand check: Δz = [-0.5, 0, -0.25] → Δx = -0.5*[1,0] -0.25*[1,1] = [-0.75, -0.25]
    expected = torch.tensor([-0.75, -0.25], dtype=torch.float32)
    assert torch.allclose(
        delta_via_matmul.float(),
        expected,
        atol=5e-3,
        rtol=1e-4,
    )
