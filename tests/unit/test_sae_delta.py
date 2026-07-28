"""SAE intervention delta tests (Phase E)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.intervention.sae_delta import normalized_coefficients


@pytest.mark.unit
def test_intervention__normalized_beta__is_scaled_featurewise() -> None:
    """SAE-001: α_j = s_j β_j featurewise."""
    scales = [2.0, 0.5]
    beta = [-1.0, -2.0]
    alpha = normalized_coefficients(scales=scales, beta=beta)
    assert alpha == pytest.approx([-2.0, -1.0], abs=1e-12, rel=1e-12)
