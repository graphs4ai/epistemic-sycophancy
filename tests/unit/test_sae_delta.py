"""SAE intervention delta tests (Phase E)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.intervention.sae_delta import (
    apply_selected_latent_update,
    normalized_coefficients,
)


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
