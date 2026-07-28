"""Coefficient regularizer tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location("golden_objective_beta", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_BETA = _golden.GOLDEN_BETA
GOLDEN_L_BETA = _golden.GOLDEN_L_BETA


@pytest.mark.unit
def test_objective__coefficient_penalty__is_mean_absolute_normalized_beta() -> None:
    """OBJ-009: L_beta = mean_j |β_j| on normalized coefficients (not raw α)."""
    from epistemic_sycophancy.objective.total import coefficient_regularizer

    l_beta = coefficient_regularizer(beta=GOLDEN_BETA)
    assert l_beta == pytest.approx(GOLDEN_L_BETA, abs=1e-12, rel=1e-12)
    assert l_beta == pytest.approx(0.5, abs=1e-12, rel=1e-12)
    # Not the sum of absolutes
    assert l_beta != pytest.approx(1.5, abs=1e-12, rel=1e-12)
