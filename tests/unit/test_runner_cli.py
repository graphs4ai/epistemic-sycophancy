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
        "full_study",
    )
    assert PIXI_TASK_NAMES == (
        "run-identity",
        "run-baseline",
        "run-fs",
        "run-opt-smoke",
        "run-study",
    )
    for stage in STAGE_ORDER[:-1]:
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
