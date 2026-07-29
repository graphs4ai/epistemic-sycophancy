"""RUN-013: staged CLI entry points and pixi task names."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.runner.cli import (
    STAGE_ORDER,
    PIXI_TASK_NAMES,
    run_stage,
)


@pytest.mark.unit
def test_runner__cli_stages__expose_identity_baseline_fs_opt_full_in_order() -> None:
    """RUN-013: stage registry order + full_study blocked without freeze."""
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
    for stage in STAGE_ORDER:
        if stage in {"full_study", "holdout_eval"}:
            continue
        result = run_stage(stage, freeze_status="unsealed")
        assert result.stage == stage
        assert result.ok is True

    with pytest.raises(HoldoutAccessError):
        run_stage("full_study", freeze_status="unsealed")

    sealed = run_stage("full_study", freeze_status="sealed")
    assert sealed.stage == "full_study"
    assert sealed.ok is True


@pytest.mark.unit
def test_runner_cli__config_path__invalid_yaml_raises_clear_validation_error(
    tmp_path,
) -> None:
    """CFGFILE-006: --config with invalid study YAML raises InvalidExperimentConfig."""
    from epistemic_sycophancy.config.schema import InvalidExperimentConfig
    from epistemic_sycophancy.runner.cli import build_arg_parser, main

    bad = tmp_path / "bad.yaml"
    bad.write_text("stack:\n  model: not-a-mapping\n", encoding="utf-8")

    parser = build_arg_parser()
    args = parser.parse_args(["identity", "--config", str(bad)])
    assert args.config == str(bad)

    with pytest.raises(InvalidExperimentConfig):
        main(["identity", "--config", str(bad)])


@pytest.mark.unit
def test_runner_cli__with_config__dispatches_real_stage_not_ready_stub(
    tmp_path, monkeypatch
) -> None:
    """WIRE-011: --config dispatches real stage functions (not 'stage … ready')."""
    from pathlib import Path

    from epistemic_sycophancy.runner import cli as cli_mod

    study_yaml = Path("configs/smokes/layer17_n2.yaml")
    calls: list[str] = []

    def fake_dispatch(stage: str, *, study, freeze_status: str, **kwargs):
        del study, freeze_status, kwargs
        calls.append(stage)
        return cli_mod.StageResult(
            stage=stage, ok=True, message=f"completed {stage}"
        )

    monkeypatch.setattr(cli_mod, "dispatch_stage", fake_dispatch)
    code = cli_mod.main(
        ["identity", "--config", str(study_yaml), "--freeze-status", "unsealed"]
    )
    assert code == 0
    assert calls == ["identity"]
