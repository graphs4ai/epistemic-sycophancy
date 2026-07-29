"""ORCH-034…038: real_model ASAP gates on layer17_n2 (replace hollow ORCH-017/018)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.runner.cli import dispatch_stage, run_cli
from epistemic_sycophancy.runner.identity import clear_stack_cache

CFG = Path("configs/smokes/layer17_n2.yaml")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.fail("CUDA required for ORCH-034…038 real_model ASAP gates (not skipped)")


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n2__identity_via_cli_default_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORCH-034: identity on layer17_n2 via default load_stack (actual residuals)."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(tmp_path / "art")))
    result = dispatch_stage(
        "identity",
        study=study,
        freeze_status="unsealed",
    )
    assert result.ok
    assert result.metrics["identity_passed"] is True
    path = Path(result.artifacts["identity_result"])
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["identity_passed"] is True
    assert float(payload["max_abs_diff"]) < 1e-5


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n2__baseline_writes_partition_without_score_fn(
    tmp_path: Path,
) -> None:
    """ORCH-035: baseline_partitions via default adapters (no score_fn kwarg)."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(tmp_path / "art")))
    result = dispatch_stage(
        "baseline_partitions",
        study=study,
        freeze_status="unsealed",
        score_fn=None,
    )
    assert result.ok
    for order in study.run.order_regimes:
        path = Path(study.run.artifact_dir) / "baseline" / f"partition_{order}.json"
        assert path.is_file(), f"missing {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "q_plus" in payload and "q_minus" in payload
        assert payload["n_q_plus"] + payload["n_q_minus"] >= 1


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n2__feature_selection_writes_pool_keys(
    tmp_path: Path,
) -> None:
    """ORCH-036: feature_selection writes common_pool with (layer, feature_id)."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudySmokeConfig

    study = load_study_config(CFG)
    study = replace(
        study,
        run=replace(
            study.run,
            artifact_dir=str(tmp_path / "art"),
            order_regimes=("CF",),
            smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
        ),
    )
    result = dispatch_stage(
        "feature_selection",
        study=study,
        freeze_status="unsealed",
        jacobian_fn=None,
        scale_fn=None,
    )
    assert result.ok
    pool_path = Path(result.artifacts["pool"])
    assert pool_path.is_file()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload["pool_size"] >= 1
    assert payload["scale_source"] == "decoder_norm"
    assert all(len(pair) == 2 for pair in payload["feature_ids"])
    assert all(s > 0 for s in payload["feature_scales"])
