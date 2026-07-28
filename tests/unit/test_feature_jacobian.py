"""Exact local coefficient Jacobian (Phase F)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from epistemic_sycophancy.feature_selection import coefficient_jacobian

_TOY_GRADIENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "feature_selection"
    / "toy_gradients.py"
)
_spec = importlib.util.spec_from_file_location(
    "toy_gradients_jacobian", _TOY_GRADIENTS_PATH
)
assert _spec is not None and _spec.loader is not None
_toy_gradients = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toy_gradients)
spec_decoder = _toy_gradients.spec_decoder
spec_gradient = _toy_gradients.spec_gradient
spec_latents = _toy_gradients.spec_latents
spec_scales = _toy_gradients.spec_scales


@pytest.mark.unit
def test_feature_jacobian__scale_and_relu_mask__match_chain_rule() -> None:
    """FEAT-004: J_j = s_j * 1[z_j > 0] * h_j.

    Hand-derived from spec §11 with h=[2,-3,1], z=[0.5,0,2], s=[2,4,0.5]:
    J = [2*1*2, 4*0*(-3), 0.5*1*1] = [4.0, 0.0, 0.5].
    """
    jacobian = coefficient_jacobian(
        raw_projection=spec_gradient() @ spec_decoder().T,
        latents=spec_latents(),
        feature_scales=spec_scales(),
    )

    assert jacobian.tolist() == [4.0, 0.0, 0.5]
