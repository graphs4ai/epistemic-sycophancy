"""ORCH-039: Phase M.1 ship docs cover adapter defaults and CUDA env."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_docs__phase_m_ship_gate__documents_adapter_defaults_and_cuda_env() -> None:
    """ORCH-039: ship gate + README document YAML-only ASAP and test-cuda."""
    ship = Path("docs/phase_m_ship_gate.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    progress = Path("docs/tdd-progress.md").read_text(encoding="utf-8")

    assert "test-cuda" in ship
    assert "layer17_n2.yaml" in ship
    assert "adapter" in ship.lower() or "without injector" in ship.lower()
    assert "score_fn" in ship or "injector" in ship.lower()
    assert "ORCH-034" in ship or "ORCH-038" in ship or "M.1" in ship
    assert "holdout" in ship.lower()
    assert "n_questions: 32" in ship or "smoke.n_questions: 32" in ship or "N=32" in ship

    assert "configs/smokes/layer17_n2.yaml" in readme
    assert "test-cuda" in readme
    assert "adapter" in readme.lower() or "YAML-only" in readme or "YAML only" in readme

    # Hollow ORCH-017/018 superseded by real M.1 gates.
    assert "ORCH-017" in progress and "superseded" in progress.lower()
    assert "ORCH-018" in progress and "superseded" in progress.lower()
    assert "ORCH-034" in progress and "ORCH-038" in progress
