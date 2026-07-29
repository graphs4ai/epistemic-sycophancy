"""Staged CLI entry points for Phase K/L experiment runs (DEC-055 / CFGFILE-006)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError

STAGE_ORDER: tuple[str, ...] = (
    "identity",
    "baseline_partitions",
    "feature_selection",
    "opt_smoke",
    "full_study",
)

PIXI_TASK_NAMES: tuple[str, ...] = (
    "run-identity",
    "run-baseline",
    "run-fs",
    "run-opt-smoke",
    "run-study",
)


@dataclass(frozen=True)
class StageResult:
    """Outcome of invoking one staged runner entry point."""

    stage: str
    ok: bool
    message: str


def run_stage(stage: str, *, freeze_status: str) -> StageResult:
    """Dispatch a named stage; block full_study until freeze is sealed."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    if stage == "full_study" and freeze_status != "sealed":
        raise HoldoutAccessError(
            "full_study requires freeze_status='sealed' "
            f"(got {freeze_status!r}; DEC-055 / DEC-042)"
        )
    return StageResult(stage=stage, ok=True, message=f"stage {stage} ready")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epistemic-sycophancy-run")
    parser.add_argument(
        "stage",
        choices=STAGE_ORDER,
        help="Experiment stage to run (DEC-055)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to StudyConfig YAML (CFGFILE-006)",
    )
    parser.add_argument(
        "--freeze-status",
        default="unsealed",
        help="FrozenExperimentConfig status (sealed required for full_study)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.config is not None:
        # Validate early; real stage dispatch lands in WIRE-011.
        load_study_config(Path(args.config))
    result = run_stage(args.stage, freeze_status=args.freeze_status)
    print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
