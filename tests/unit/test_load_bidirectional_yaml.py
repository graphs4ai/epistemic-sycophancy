"""CFGFILE: bidirectional study YAML loads with coefficient_mode and ±2 bounds."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_load_study__bidirectional_yaml__sets_mode_and_bounds() -> None:
    """CFGFILE-BIDIR / DEC-105: sibling YAML enables bidirectional mining/opt."""
    from epistemic_sycophancy.config.load_study import load_study_config

    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "gemma3_4b_resid_post_65k_medium_l17l22_CF_bidirectional.yaml"
    )
    study = load_study_config(path)
    assert study.experiment.coefficient_mode == "bidirectional"
    assert study.experiment.beta_lower == pytest.approx(-2.0)
    assert study.experiment.beta_upper == pytest.approx(2.0)
    assert "bidirectional" in study.run.artifact_dir
    assert study.experiment.pool_quota_per_list == 64
