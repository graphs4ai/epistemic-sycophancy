"""Staged CLI entry points for Phase K/L/M experiment runs (DEC-055 / DEC-072)."""

from __future__ import annotations

import argparse
import json
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
    "optimize",
    "freeze",
    "full_study",
    "holdout_eval",
)

PIXI_TASK_NAMES: tuple[str, ...] = (
    "run-identity",
    "run-baseline",
    "run-fs",
    "run-opt-smoke",
    "run-optimize",
    "run-freeze",
    "run-study",
    "run-holdout",
)


@dataclass(frozen=True)
class StageResult:
    """Outcome of invoking one staged runner entry point (ORCH-006)."""

    stage: str
    ok: bool
    message: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    study_yaml_fingerprint: str = ""
    model_revision: str = ""
    sae_revision: str = ""
    hook_configuration_hash: str = ""
    layer_set_hash: str = ""


def _stage_hash_fields(study: StudyConfig) -> dict[str, str]:
    """Populate StageResult hash fields from StudyConfig (WIRE-012 / ORCH-006)."""
    import json
    from hashlib import sha256

    from epistemic_sycophancy.config.load_study import study_config_fingerprint

    hook_payload = {
        "token_scope": study.stack.hooks.token_scope,
        "resolver_id": study.stack.hooks.resolver_id,
        "k": study.stack.hooks.k,
    }
    hook_configuration_hash = sha256(
        json.dumps(hook_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    layer_set_hash = sha256(
        ",".join(str(layer) for layer in study.stack.sae.layers).encode()
    ).hexdigest()
    return {
        "study_yaml_fingerprint": study_config_fingerprint(study),
        "model_revision": study.stack.model.revision,
        "sae_revision": (
            f"{study.stack.sae.release}:{study.stack.sae.width}:{study.stack.sae.l0}"
        ),
        "hook_configuration_hash": hook_configuration_hash,
        "layer_set_hash": layer_set_hash,
    }


def _make_result(
    *,
    stage: str,
    ok: bool,
    message: str,
    study: StudyConfig,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> StageResult:
    hashes = _stage_hash_fields(study)
    return StageResult(
        stage=stage,
        ok=ok,
        message=message,
        metrics=dict(metrics or {}),
        artifacts=dict(artifacts or {}),
        study_yaml_fingerprint=hashes["study_yaml_fingerprint"],
        model_revision=hashes["model_revision"],
        sae_revision=hashes["sae_revision"],
        hook_configuration_hash=hashes["hook_configuration_hash"],
        layer_set_hash=hashes["layer_set_hash"],
    )


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
    objective_fn: Callable[..., Any] | None = None,
    grad_fn: Callable[..., Any] | None = None,
    eval_payload: Mapping[str, Any] | None = None,
    frozen_config_path: str | None = None,
    holdout_rows_provider: Callable[[], Any] | None = None,
    corpus_jsonl_paths: Sequence[str | Path] | None = None,
    split_manifest_path: str | Path | None = None,
    corpus_root: str | Path | None = None,
) -> StageResult:
    """Dispatch a real stage using validated StudyConfig (WIRE-011 / ORCH-001)."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")

    from epistemic_sycophancy.config.load_study import study_config_fingerprint

    if stage == "full_study":
        if freeze_status != "sealed":
            raise HoldoutAccessError(
                "full_study requires freeze_status='sealed' "
                f"(got {freeze_status!r}; DEC-055 / DEC-042 / DEC-063)"
            )
        from epistemic_sycophancy.runner.adapters.eval_payload import build_eval_payload
        from epistemic_sycophancy.runner.full_study import run_full_study_dispatch
        from epistemic_sycophancy.runner.identity import resolve_stack
        from epistemic_sycophancy.optimization.checkpoint import load_checkpoint

        if eval_payload is None:
            stack = resolve_stack(study, stack_loader=stack_loader)
            ckpt_path = Path(study.run.artifact_dir) / "optimize" / "best_checkpoint.json"
            ckpt = load_checkpoint(json.loads(ckpt_path.read_text(encoding="utf-8")))
            best_beta = tuple(float(x) for x in ckpt["beta"])
            val_ids = tuple(validation_question_ids or ())
            if not val_ids:
                from epistemic_sycophancy.runner.adapters.resolve import (
                    resolve_corpus_context,
                )

                _corpus, split_ids, _smoke = resolve_corpus_context(
                    study,
                    corpus_jsonl_paths=corpus_jsonl_paths,
                    split_manifest_path=split_manifest_path,
                    corpus_root=corpus_root,
                )
                del _corpus, _smoke
                val_ids = tuple(split_ids.get("behavior_validation", ()))
            eval_payload = build_eval_payload(
                study,
                stack,
                best_beta=best_beta,
                validation_question_ids=val_ids,
                margin_scorer=getattr(stack, "score_belief_margins", None),
                holdout_question_ids=holdout_question_ids or (),
            )
        fs = run_full_study_dispatch(
            study=study,
            freeze_status=freeze_status,
            eval_payload=eval_payload,
            holdout_question_ids=holdout_question_ids or (),
        )
        return _make_result(
            stage=stage,
            ok=True,
            message=(
                f"completed full_study: n_cells={fs['metrics']['n_cells']} "
                f"study_fp={study_config_fingerprint(study)[:12]}…"
            ),
            study=study,
            metrics=dict(fs["metrics"]),
            artifacts=dict(fs["artifacts"]),
        )

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
        return _make_result(
            stage=stage,
            ok=ok,
            message=message,
            study=study,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "baseline_partitions":
        from epistemic_sycophancy.runner.baseline import run_baseline_dispatch
        from epistemic_sycophancy.runner.identity import resolve_stack

        regimes = tuple(str(o) for o in study.run.order_regimes) or ("CF",)

        if score_fn is None:
            from epistemic_sycophancy.runner.adapters.resolve import resolve_corpus_context
            from epistemic_sycophancy.runner.adapters.score import build_score_fn

            stack = resolve_stack(study, stack_loader=stack_loader)
            corpus, split_ids, smoke_ids = resolve_corpus_context(
                study,
                corpus_jsonl_paths=corpus_jsonl_paths,
                split_manifest_path=split_manifest_path,
                corpus_root=corpus_root,
            )
            artifacts: dict[str, str] = {}
            metrics: dict[str, Any] = {"order_regimes": list(regimes)}
            last_metrics: dict[str, Any] = {}
            for order in regimes:
                one = run_baseline_dispatch(
                    study=study,
                    freeze_status=freeze_status,
                    score_fn=build_score_fn(
                        study,
                        stack,
                        corpus=corpus,
                        split_question_ids=split_ids,
                        order_regime=order,
                        belief_condition="N",
                    ),
                    question_ids=smoke_ids,
                    split_name=split_name_override,
                    order_regimes=(order,),
                )
                artifacts.update(one["artifacts"])
                last_metrics = dict(one["metrics"])
                metrics[f"n_q_plus_{order}"] = one["metrics"]["n_q_plus"]
                metrics[f"n_q_minus_{order}"] = one["metrics"]["n_q_minus"]
            metrics.update(
                {
                    "n_q_plus": last_metrics["n_q_plus"],
                    "n_q_minus": last_metrics["n_q_minus"],
                    "n_q_tie": last_metrics.get("n_q_tie", 0),
                    "order_regime": last_metrics.get("order_regime", regimes[-1]),
                    "q_plus": last_metrics.get("q_plus", []),
                    "q_minus": last_metrics.get("q_minus", []),
                }
            )
            if "partition" not in artifacts and artifacts:
                artifacts["partition"] = next(iter(artifacts.values()))
            message = (
                f"completed baseline_partitions: n_q_plus={metrics['n_q_plus']} "
                f"n_q_minus={metrics['n_q_minus']} "
                f"study_fp={fingerprint[:12]}…"
            )
            return _make_result(
                stage=stage,
                ok=True,
                message=message,
                study=study,
                metrics=metrics,
                artifacts=artifacts,
            )

        baseline = run_baseline_dispatch(
            study=study,
            freeze_status=freeze_status,
            score_fn=score_fn,
            split_name=split_name_override,
            order_regimes=regimes,
        )
        metrics = dict(baseline["metrics"])
        artifacts = dict(baseline["artifacts"])
        message = (
            f"completed baseline_partitions: n_q_plus={metrics['n_q_plus']} "
            f"n_q_minus={metrics['n_q_minus']} "
            f"study_fp={fingerprint[:12]}…"
        )
        return _make_result(
            stage=stage,
            ok=True,
            message=message,
            study=study,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "feature_selection":
        from epistemic_sycophancy.runner.adapters.jacobian import build_jacobian_fn
        from epistemic_sycophancy.runner.adapters.resolve import resolve_corpus_context
        from epistemic_sycophancy.runner.adapters.scales import build_scale_fn
        from epistemic_sycophancy.runner.fs_dispatch import run_feature_selection_dispatch
        from epistemic_sycophancy.runner.identity import resolve_stack

        resolved_qids = None
        if jacobian_fn is None or scale_fn is None:
            stack = resolve_stack(study, stack_loader=stack_loader)
            _corpus, _split_ids, smoke_ids = resolve_corpus_context(
                study,
                corpus_jsonl_paths=corpus_jsonl_paths,
                split_manifest_path=split_manifest_path,
                corpus_root=corpus_root,
            )
            del _corpus, _split_ids
            resolved_qids = smoke_ids
            if jacobian_fn is None:
                jacobian_fn = build_jacobian_fn(study, stack)
            if scale_fn is None:
                scale_fn = build_scale_fn(study, stack)

        fs = run_feature_selection_dispatch(
            study=study,
            freeze_status=freeze_status,
            jacobian_fn=jacobian_fn,
            scale_fn=scale_fn,
            question_ids=resolved_qids,
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
        return _make_result(
            stage=stage,
            ok=True,
            message=message,
            study=study,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "opt_smoke":
        from epistemic_sycophancy.runner.adapters.identity_gate import (
            resolve_identity_passed,
        )
        from epistemic_sycophancy.runner.adapters.margins import build_margin_payload
        from epistemic_sycophancy.runner.adapters.pool import (
            load_common_pool_artifact,
            study_with_selected_pool,
        )
        from epistemic_sycophancy.runner.identity import resolve_stack
        from epistemic_sycophancy.runner.opt_smoke_dispatch import run_opt_smoke_dispatch

        study_for_opt = study
        if study.experiment.coefficient_length < 1:
            pool_path = (
                Path(study.run.artifact_dir) / "feature_selection" / "common_pool.json"
            )
            pool = load_common_pool_artifact(pool_path)
            study_for_opt = study_with_selected_pool(study, pool)

        if identity_passed is None:
            identity_passed = resolve_identity_passed(study_for_opt)

        if margin_payload is None or beta is None:
            stack = resolve_stack(study_for_opt, stack_loader=stack_loader)
            m = int(study_for_opt.experiment.coefficient_length)
            if m < 1:
                raise ValueError("opt_smoke requires selected pool (coefficient_length>=1)")
            if beta is None:
                beta = tuple(0.0 for _ in range(m))
            if margin_payload is None:
                part_path = (
                    Path(study.run.artifact_dir) / "baseline" / "partition_CF.json"
                )
                if not part_path.is_file():
                    raise ValueError(
                        f"opt_smoke default adapters require baseline partition at {part_path}"
                    )
                part = json.loads(part_path.read_text(encoding="utf-8"))
                partitions = {
                    "q_plus": frozenset(part["q_plus"]),
                    "q_minus": frozenset(part["q_minus"]),
                }
                smoke_ids = (
                    tuple(study.run.smoke.question_ids)
                    if study.run.smoke.question_ids is not None
                    else tuple(sorted(partitions["q_plus"] | partitions["q_minus"]))
                )
                margin_payload = build_margin_payload(
                    study_for_opt,
                    stack,
                    beta=beta,
                    question_ids=smoke_ids,
                    partitions=partitions,
                )

        smoke = run_opt_smoke_dispatch(
            study=study_for_opt,
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
        return _make_result(
            stage=stage,
            ok=True,
            message=message,
            study=study,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "optimize":
        from epistemic_sycophancy.runner.adapters.identity_gate import (
            resolve_identity_passed,
        )
        from epistemic_sycophancy.runner.adapters.objective import (
            build_grad_fn,
            build_objective_fn,
        )
        from epistemic_sycophancy.runner.adapters.pool import (
            load_common_pool_artifact,
            study_with_selected_pool,
        )
        from epistemic_sycophancy.runner.adapters.resolve import resolve_corpus_context
        from epistemic_sycophancy.runner.identity import resolve_stack
        from epistemic_sycophancy.runner.optimize import run_optimize_dispatch

        study_for_opt = study
        if study.experiment.coefficient_length < 1:
            pool_path = (
                Path(study.run.artifact_dir) / "feature_selection" / "common_pool.json"
            )
            pool = load_common_pool_artifact(pool_path)
            study_for_opt = study_with_selected_pool(study, pool)

        if identity_passed is None:
            identity_passed = resolve_identity_passed(study_for_opt)

        opt_qids = tuple(optimization_question_ids or ())
        if not opt_qids:
            _corpus, split_ids, _smoke = resolve_corpus_context(
                study,
                corpus_jsonl_paths=corpus_jsonl_paths,
                split_manifest_path=split_manifest_path,
                corpus_root=corpus_root,
            )
            del _corpus, _smoke
            from epistemic_sycophancy.runner.adapters.corpus import (
                resolve_optimize_coverage_ids,
            )

            opt_qids = resolve_optimize_coverage_ids(
                optimize=study.run.optimize,
                split_question_ids=split_ids,
            )

        if objective_fn is None or (
            grad_fn is None and study.run.optimizer.kind == "projected_adam"
        ):
            stack = resolve_stack(study_for_opt, stack_loader=stack_loader)
            part_path = Path(study.run.artifact_dir) / "baseline" / "partition_CF.json"
            partitions: dict[str, Any]
            if part_path.is_file():
                part = json.loads(part_path.read_text(encoding="utf-8"))
                artifact_plus = frozenset(str(q) for q in part["q_plus"])
                artifact_minus = frozenset(str(q) for q in part["q_minus"])
                eligible_set = frozenset(str(q) for q in opt_qids)
                inter_plus = artifact_plus & eligible_set
                inter_minus = artifact_minus & eligible_set
                if inter_plus and inter_minus:
                    partitions = {"q_plus": inter_plus, "q_minus": inter_minus}
                else:
                    # Eligible optimize IDs disjoint from FS baseline: score neutrals
                    # at β=0 on eligible set for objective partitions (DEC-076 live).
                    from epistemic_sycophancy.metrics.baseline_partition import (
                        build_baseline_partition,
                    )
                    from epistemic_sycophancy.runner.adapters.margins import (
                        build_margin_payload,
                    )

                    m = int(study_for_opt.experiment.coefficient_length)
                    zero = tuple(0.0 for _ in range(m)) if m else (0.0,)
                    # Temporary partitions so build_margin_payload can run; replace below.
                    tmp = {
                        "q_plus": frozenset(opt_qids[:1] or opt_qids),
                        "q_minus": frozenset(opt_qids[1:2] or opt_qids),
                    }
                    payload0 = build_margin_payload(
                        study_for_opt,
                        stack,
                        beta=zero,
                        question_ids=opt_qids,
                        partitions=tmp,
                    )
                    built = build_baseline_partition(
                        order_regime="CF",
                        neutral_margins=payload0["baseline_neutral_margins"],
                        epsilon=float(study.experiment.tie_band_epsilon),
                        tie_policy=str(study.experiment.tie_policy),
                    )
                    partitions = {
                        "q_plus": frozenset(built.q_plus),
                        "q_minus": frozenset(built.q_minus),
                    }
            else:
                raise ValueError(
                    f"optimize default adapters require baseline partition at {part_path}"
                )
            if objective_fn is None:
                objective_fn = build_objective_fn(
                    study_for_opt,
                    stack,
                    partitions=partitions,
                )
            if grad_fn is None and study.run.optimizer.kind == "projected_adam":
                grad_fn = build_grad_fn(
                    study_for_opt,
                    stack,
                    partitions=partitions,
                )

        opt_result = run_optimize_dispatch(
            study=study_for_opt,
            freeze_status=freeze_status,
            identity_passed=bool(identity_passed),
            optimization_question_ids=opt_qids,
            objective_fn=objective_fn,
            grad_fn=grad_fn,
            beta_init=beta,
        )
        metrics = dict(opt_result["metrics"])
        artifacts = dict(opt_result["artifacts"])
        message = (
            f"completed optimize: kind={metrics['optimizer_kind']} "
            f"n_trials={metrics['n_trials']} "
            f"study_fp={fingerprint[:12]}…"
        )
        return _make_result(
            stage=stage,
            ok=True,
            message=message,
            study=study,
            metrics=metrics,
            artifacts=artifacts,
        )

    if stage == "freeze":
        from epistemic_sycophancy.runner.freeze_stage import run_freeze_dispatch

        frozen = run_freeze_dispatch(study=study)
        return _make_result(
            stage=stage,
            ok=True,
            message=(
                f"completed freeze: freeze_status=sealed "
                f"study_fp={fingerprint[:12]}…"
            ),
            study=study,
            metrics=dict(frozen["metrics"]),
            artifacts=dict(frozen["artifacts"]),
        )

    if stage == "holdout_eval":
        from epistemic_sycophancy.runner.holdout_eval import run_holdout_eval_dispatch

        if frozen_config_path is None or holdout_rows_provider is None:
            raise ValueError(
                "holdout_eval requires frozen_config_path and holdout_rows_provider "
                "(ORCH-015)"
            )
        holdout = run_holdout_eval_dispatch(
            study=study,
            freeze_status=freeze_status,
            frozen_config_path=frozen_config_path,
            holdout_rows_provider=holdout_rows_provider,
        )
        return _make_result(
            stage=stage,
            ok=True,
            message=(
                f"completed holdout_eval: n_rows={holdout['metrics']['n_holdout_rows']} "
                f"study_fp={fingerprint[:12]}…"
            ),
            study=study,
            metrics=dict(holdout["metrics"]),
            artifacts=dict(holdout["artifacts"]),
        )

    return _make_result(
        stage=stage,
        ok=True,
        message=f"completed {stage}",
        study=study,
    )


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


def run_cli(
    argv: list[str] | None = None,
    *,
    stack_loader: Callable[[StudyConfig], Any] | None = None,
    **dispatch_kwargs: Any,
) -> int:
    """Testable CLI entry: load StudyConfig and dispatch with optional injections."""
    args = build_arg_parser().parse_args(argv)
    if args.config is None:
        raise SystemExit(
            "error: --config PATH is required for real stage dispatch (WIRE-011)"
        )
    study = load_study_config(Path(args.config))
    result = dispatch_stage(
        args.stage,
        study=study,
        freeze_status=args.freeze_status,
        stack_loader=stack_loader,
        **dispatch_kwargs,
    )
    print(result.message)
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
