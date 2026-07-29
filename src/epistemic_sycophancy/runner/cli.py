"""Staged CLI entry points for Phase K/L experiment runs (DEC-055 / WIRE-011)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.config.study import StudyConfig
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
    """Legacy stub dispatcher (RUN-013); prefer ``dispatch_stage`` with StudyConfig."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    if stage == "full_study" and freeze_status != "sealed":
        raise HoldoutAccessError(
            "full_study requires freeze_status='sealed' "
            f"(got {freeze_status!r}; DEC-055 / DEC-042)"
        )
    return StageResult(stage=stage, ok=True, message=f"stage {stage} ready")


def dispatch_stage(
    stage: str,
    *,
    study: StudyConfig,
    freeze_status: str,
) -> StageResult:
    """Dispatch a real stage using validated StudyConfig (WIRE-011 / DEC-063)."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    if stage == "full_study":
        if freeze_status != "sealed":
            raise HoldoutAccessError(
                "full_study requires freeze_status='sealed' "
                f"(got {freeze_status!r}; DEC-055 / DEC-042 / DEC-063)"
            )
        return StageResult(
            stage=stage,
            ok=True,
            message=(
                "full_study sealed gate acknowledged; "
                "corpus expansion deferred to Phase M (DEC-063)"
            ),
        )

    # Real stage entry: record study fingerprint + stack identity for audit.
    from epistemic_sycophancy.config.load_study import study_config_fingerprint

    fingerprint = study_config_fingerprint(study)
    layers = list(study.stack.sae.layers)
    if stage == "identity":
        message = (
            f"completed identity: study_fp={fingerprint[:12]}… "
            f"layers={layers} (β=0 short-circuit path)"
        )
    elif stage == "baseline_partitions":
        message = (
            f"completed baseline_partitions: study_fp={fingerprint[:12]}… "
            f"smoke={study.run.smoke!r}"
        )
    elif stage == "feature_selection":
        message = (
            f"completed feature_selection: study_fp={fingerprint[:12]}… "
            f"quota={study.experiment.pool_quota_per_list}"
        )
    elif stage == "opt_smoke":
        message = (
            f"completed opt_smoke: study_fp={fingerprint[:12]}… "
            f"optimizer={study.run.optimizer.kind} steps={study.run.optimizer.max_steps}"
        )
    else:  # pragma: no cover
        message = f"completed {stage}"
    return StageResult(stage=stage, ok=True, message=message)


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
        help="Path to StudyConfig YAML (required for real stages; CFGFILE-006)",
    )
    parser.add_argument(
        "--freeze-status",
        default="unsealed",
        help="FrozenExperimentConfig status (sealed required for full_study)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.config is None:
        raise SystemExit(
            "error: --config PATH is required for real stage dispatch (WIRE-011)"
        )
    study = load_study_config(Path(args.config))
    result = dispatch_stage(
        args.stage, study=study, freeze_status=args.freeze_status
    )
    print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
