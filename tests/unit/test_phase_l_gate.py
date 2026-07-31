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
_DEV_LIMITED = _REPO_ROOT / "configs" / "dev" / "layer17_n32.yaml"


@pytest.mark.unit
def test_phase_l_gate__yaml_to_finite_objective_contract_documented() -> None:
    """WIRE-013: Study YAML loads; stage order pinned; full_study stays sealed.

    Live stage dispatch without injectors is Phase M.1 (ORCH-033+ / ORCH-014).
    This gate only documents the Phase L YAML + holdout-seal contract (DEC-063).
    """
    study = load_study_config(_FIRST_STUDY)
    assert study.stack.sae.layers == (9, 17, 22, 29)
    dev = load_study_config(_DEV_LIMITED)
    assert dev.stack.sae.layers == (17,)
    assert dev.run.fs_coverage.n_questions == 32

    assert STAGE_ORDER[:4] == (
        "identity",
        "baseline_partitions",
        "feature_selection",
        "optimize",
    )
    assert STAGE_ORDER[-2:] == ("full_study", "holdout_eval")
    assert callable(dispatch_stage)

    with pytest.raises(HoldoutAccessError):
        dispatch_stage("full_study", study=study, freeze_status="unsealed")
