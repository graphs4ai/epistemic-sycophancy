"""GRAD-009: ship gate docs record GRAD-FIX re-run and flat-trials failure."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_docs__phase_m_ship_gate__documents_grad_fix_rerun_and_flat_failure() -> None:
    """GRAD-009: phase_m_ship_gate must not treat flat all-zero trials as success."""
    ship = Path("docs/phase_m_ship_gate.md").read_text(encoding="utf-8")
    assert "GRAD-FIX" in ship or "DEC-084" in ship
    assert "GRAD-008" in ship
    assert "all-zero" in ship.casefold() or "flat" in ship.casefold()
    assert "re-run" in ship.casefold() or "rerun" in ship.casefold()
    progress = Path("docs/tdd-progress.md").read_text(encoding="utf-8")
    assert "GRAD-008" in progress and "GRAD-007" in progress
