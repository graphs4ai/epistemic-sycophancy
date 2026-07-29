"""Phase L gate: StudyConfig YAML + sealed full_study contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.runner.cli import STAGE_ORDER, dispatch_stage


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIRST_STUDY = (
    _REPO_ROOT / "configs" / "first_study_gemma3_4b_resid_post_65k_medium.yaml"
)
_SMOKE = _REPO_ROOT / "configs" / "smokes" / "layer17_n2.yaml"


@pytest.mark.unit
def test_phase_l_gate__yaml_to_finite_objective_contract_documented() -> None:
    """WIRE-013: Study YAML loads; stage entrypoints exist; full_study stays sealed."""
    study = load_study_config(_FIRST_STUDY)
    assert study.stack.sae.layers == (9, 17, 22, 29)
    smoke = load_study_config(_SMOKE)
    assert smoke.stack.sae.layers == (17,)

    assert STAGE_ORDER[:4] == (
        "identity",
        "baseline_partitions",
        "feature_selection",
        "opt_smoke",
    )
    for stage in STAGE_ORDER[:4]:
        result = dispatch_stage(stage, study=smoke, freeze_status="unsealed")
        assert result.ok
        assert "ready" not in result.message or "completed" in result.message
        assert result.message.startswith("completed ")

    with pytest.raises(HoldoutAccessError):
        dispatch_stage("full_study", study=study, freeze_status="unsealed")

    sealed = dispatch_stage("full_study", study=study, freeze_status="sealed")
    assert "Phase M" in sealed.message
