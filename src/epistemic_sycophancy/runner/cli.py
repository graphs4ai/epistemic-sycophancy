"""Staged CLI entry points for Phase K/L/M experiment runs (DEC-055 / DEC-072)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)


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
    stack_loader: Callable[[StudyConfig], Any] | None = None,
    score_fn: Callable[..., Any] | None = None,
    jacobian_fn: Callable[..., Any] | None = None,
    scale_fn: Callable[..., Any] | None = None,
    split_name_override: str | None = None,
    optimization_question_ids: Sequence[str] | None = None,
    validation_question_ids: Sequence[str] | None = None,
    holdout_question_ids: Sequence[str] | None = None,
    identity_passed: bool | None = None,
    margin_payload: Mapping[str, Any] | None = None,
    beta: Sequence[float] | None = None,
    adam_grad: Sequence[float] | None = None,
) -> StageResult:
    """Dispatch a real stage using validated StudyConfig (WIRE-011 / ORCH-001)."""
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

    from epistemic_sycophancy.config.load_study import study_config_fingerprint

    fingerprint = study_config_fingerprint(study)
    layers = list(study.stack.sae.layers)

    if stage == "identity":
        from epistemic_sycophancy.runner.identity import resolve_stack, run_identity_stage

        stack = resolve_stack(study, stack_loader=stack_loader)
        identity = run_identity_stage(study=study, stack=stack)
        ok = bool(identity["identity_passed"])
        metrics = {
            "identity_passed": identity["identity_passed"],
            "max_abs_diff": identity["max_abs_diff"],
        }
        artifacts = dict(identity["artifacts"])
        message = (
            f"completed identity: identity_passed={ok} "
            f"max_abs_diff={identity['max_abs_diff']} "
            f"layers={layers} study_fp={fingerprint[:12]}…"
        )
        return StageResult(
            stage=stage,
            ok=ok,
            message=message,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "baseline_partitions":
        from epistemic_sycophancy.runner.baseline import run_baseline_dispatch

        if score_fn is None:
            raise ValueError(
                "baseline_partitions requires score_fn injection or stack scoring "
                "(ORCH-003); pass score_fn for unit tests"
            )
        baseline = run_baseline_dispatch(
            study=study,
            freeze_status=freeze_status,
            score_fn=score_fn,
            split_name=split_name_override,
        )
        metrics = dict(baseline["metrics"])
        artifacts = dict(baseline["artifacts"])
        message = (
            f"completed baseline_partitions: n_q_plus={metrics['n_q_plus']} "
            f"n_q_minus={metrics['n_q_minus']} "
            f"study_fp={fingerprint[:12]}…"
        )
        return StageResult(
            stage=stage,
            ok=True,
            message=message,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "feature_selection":
        from epistemic_sycophancy.runner.fs_dispatch import run_feature_selection_dispatch

        if jacobian_fn is None or scale_fn is None:
            raise ValueError(
                "feature_selection requires jacobian_fn and scale_fn injection "
                "(ORCH-004)"
            )
        fs = run_feature_selection_dispatch(
            study=study,
            freeze_status=freeze_status,
            jacobian_fn=jacobian_fn,
            scale_fn=scale_fn,
            optimization_question_ids=optimization_question_ids or (),
            validation_question_ids=validation_question_ids or (),
            holdout_question_ids=holdout_question_ids or (),
        )
        metrics = dict(fs["metrics"])
        artifacts = dict(fs["artifacts"])
        message = (
            f"completed feature_selection: pool_size={metrics['pool_size']} "
            f"scale_source={metrics['scale_source']} "
            f"study_fp={fingerprint[:12]}…"
        )
        return StageResult(
            stage=stage,
            ok=True,
            message=message,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "opt_smoke":
        from epistemic_sycophancy.runner.opt_smoke_dispatch import run_opt_smoke_dispatch

        if margin_payload is None or beta is None:
            raise ValueError(
                "opt_smoke requires margin_payload and beta (ORCH-005)"
            )
        if identity_passed is None:
            raise ValueError("opt_smoke requires identity_passed (ORCH-005)")
        smoke = run_opt_smoke_dispatch(
            study=study,
            freeze_status=freeze_status,
            identity_passed=bool(identity_passed),
            margin_payload=margin_payload,
            beta=beta,
            adam_grad=adam_grad,
        )
        metrics = dict(smoke["metrics"])
        artifacts = dict(smoke["artifacts"])
        message = (
            f"completed opt_smoke: l_total={metrics['l_total']} "
            f"optimizer={study.run.optimizer.kind} "
            f"steps={study.run.optimizer.max_steps} "
            f"study_fp={fingerprint[:12]}…"
        )
        return StageResult(
            stage=stage,
            ok=True,
            message=message,
            metrics=metrics,
            artifacts=artifacts,
        )

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
