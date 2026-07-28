"""Property tests for SAE latent intervention (Phase E)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.intervention.sae_delta import apply_selected_latent_update


@pytest.mark.property
@given(alpha=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_intervention__nonpositive_alpha__cannot_activate_zero_latent(
    alpha: float,
) -> None:
    """SAE-004: z_j == 0 and α_j <= 0 ⇒ z'_j == 0."""
    updated = apply_selected_latent_update(
        latents=[0.0],
        selected_indices=[0],
        scales=[1.0],
        beta=[alpha],
    )
    assert updated[0] == 0.0
