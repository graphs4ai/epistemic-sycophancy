"""ORCH-016: README + pixi tasks cover full Phase M stage list."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.runner.cli import PIXI_TASK_NAMES, STAGE_ORDER


@pytest.mark.unit
def test_phase_m_docs__pixi_tasks_and_readme_cover_full_stage_list() -> None:
    """ORCH-016: pixi tasks + README document full stage path; deprecate ready stub."""
    assert STAGE_ORDER == (
        "identity",
        "baseline_partitions",
        "feature_selection",
        "opt_smoke",
        "optimize",
        "freeze",
        "full_study",
        "holdout_eval",
    )
    assert PIXI_TASK_NAMES == (
        "run-identity",
        "run-baseline",
        "run-fs",
        "run-opt-smoke",
        "run-optimize",
        "run-freeze",
        "run-study",
        "run-holdout",
    )
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    for task in PIXI_TASK_NAMES:
        assert f"{task} =" in pyproject or f'{task} =' in pyproject
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Phase M" in readme
    assert "optimize" in readme
    assert "freeze" in readme
    assert "holdout_eval" in readme or "run-holdout" in readme
    assert "stage ready" not in readme.lower() or "deprecated" in readme.lower()
    assert "run.optimize" in readme or "non-smoke" in readme
