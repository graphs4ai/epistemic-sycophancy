"""Optimize stage (ORCH-009 / DEC-066 / DEC-072 / DEC-097)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.logging.loss_curve import plot_loss_over_trials
from epistemic_sycophancy.logging.optimize_metrics import (
    ITERATION_CSV_COLUMNS,
    STEP_CSV_COLUMNS,
    count_betas_at_bounds,
    plot_iteration_metric_curves,
    write_optimize_metrics_csv,
)
from epistemic_sycophancy.logging.pipeline import log_progress
from epistemic_sycophancy.objective.total import ObjectiveResult
from epistemic_sycophancy.optimization.checkpoint import dump_checkpoint
from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam
from epistemic_sycophancy.reproducibility.phase_gates import require_identity_gate
from epistemic_sycophancy.runner.progress import adam_step_batch_progress

_LOSS_COMPONENT_KEYS: tuple[str, ...] = (
    "l_resist",
    "l_recover",
    "l_behavior",
    "l_neutral",
    "l_correct",
    "l_beta",
    "l_total",
)


def coerce_objective(
    value: float | ObjectiveResult | Mapping[str, Any],
) -> tuple[float, dict[str, float | None]]:
    """Normalize float / ObjectiveResult / mapping into ``(l_total, components)``."""
    if isinstance(value, ObjectiveResult):
        comps: dict[str, float | None] = {
            "l_resist": float(value.l_resist),
            "l_recover": float(value.l_recover),
            "l_behavior": float(value.l_behavior),
            "l_neutral": float(value.l_neutral),
            "l_correct": float(value.l_correct),
            "l_beta": float(value.l_beta),
            "l_total": float(value.l_total),
        }
        return float(value.l_total), comps
    if isinstance(value, Mapping):
        comps = {key: None for key in _LOSS_COMPONENT_KEYS}
        for key in _LOSS_COMPONENT_KEYS:
            if key in value and value[key] is not None:
                comps[key] = float(value[key])
        if comps["l_total"] is None:
            raise ValueError("objective mapping must include l_total")
        return float(comps["l_total"]), comps
    loss = float(value)
    comps = {key: None for key in _LOSS_COMPONENT_KEYS}
    comps["l_total"] = loss
    return loss, comps


def _metric_row(
    *,
    index: int,
    optimizer_kind: str,
    comps: Mapping[str, float | None],
    beta: Sequence[float],
    beta_lower: float,
    beta_upper: float,
    step_grad_norm: float | None = None,
) -> dict[str, Any]:
    n_lo, n_hi = count_betas_at_bounds(
        beta, beta_lower=beta_lower, beta_upper=beta_upper
    )
    row: dict[str, Any] = {
        "index": int(index),
        "optimizer_kind": str(optimizer_kind),
        "number_at_lower_bound": int(n_lo),
        "number_at_upper_bound": int(n_hi),
    }
    for key in _LOSS_COMPONENT_KEYS:
        value = comps.get(key)
        row[key] = "" if value is None else float(value)
    if step_grad_norm is not None:
        row["step_grad_norm"] = float(step_grad_norm)
    return row


def resolve_optimize_question_ids(
    *,
    study: StudyConfig,
    optimization_question_ids: Sequence[str],
) -> tuple[str, ...]:
    """Resolve eligible optimize IDs (DEC-068): explicit xor n_questions xor full split."""
    opt = study.run.optimize
    pool = tuple(str(q) for q in optimization_question_ids)
    if opt.question_ids is not None:
        return tuple(opt.question_ids)
    if opt.n_questions is not None:
        sorted_ids = sorted(pool)
        return tuple(sorted_ids[: int(opt.n_questions)])
    return pool


def run_optimize_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    identity_passed: bool,
    optimization_question_ids: Sequence[str],
    objective_fn: Callable[[Sequence[float], Sequence[str]], Any],
    grad_fn: Callable[[Sequence[float], Sequence[str]], Sequence[float]] | None = None,
    beta_init: Sequence[float] | None = None,
    adam_step_batch_total: int | None = None,
    n_q_plus: int | None = None,
    n_q_minus: int | None = None,
) -> dict[str, Any]:
    """Run optimize using ``run.optimize`` budgets.

    ``objective_fn`` may return a float, ``ObjectiveResult``, or a mapping with
    ``l_*`` keys (DEC-097). ``adam_step_batch_total`` is the fixed prompt-microbatch
    count per Adam step (DEC-092); when omitted for projected Adam, each step bar
    uses total 0.

    ``n_q_plus`` / ``n_q_minus`` are written once to ``optimize/static.json`` when
    provided (eligible ∩ frozen partition sizes).
    """
    del freeze_status
    require_identity_gate(identity_passed=identity_passed)
    eligible = resolve_optimize_question_ids(
        study=study,
        optimization_question_ids=optimization_question_ids,
    )
    if not eligible:
        raise ValueError("optimize eligible question set is empty")

    exp = study.experiment
    m = int(exp.coefficient_length)
    if m < 1:
        raise ValueError("optimize requires coefficient_length >= 1 (selected pool)")
    beta0 = list(beta_init) if beta_init is not None else [0.0] * m
    if len(beta0) != m:
        raise ValueError(f"beta_init length {len(beta0)} != coefficient_length {m}")

    kind = study.run.optimizer.kind
    optimize = study.run.optimize
    out_dir = Path(study.run.artifact_dir) / "optimize"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    best_beta = list(beta0)
    best_loss = float("inf")
    stopped_early = False
    patience = optimize.patience

    if kind == "projected_adam":
        import torch

        if optimize.max_steps is None:
            raise ValueError("projected_adam optimize requires run.optimize.max_steps")
        if grad_fn is None:
            raise ValueError("projected_adam optimize requires grad_fn")
        beta_param = torch.nn.Parameter(
            torch.tensor(beta0, dtype=torch.float64)
        )
        adam = ProjectedAdam(
            beta=beta_param,
            adam_lr=float(study.run.optimizer.adam_lr),
            adam_beta1=float(study.run.optimizer.adam_beta1),
            adam_beta2=float(study.run.optimizer.adam_beta2),
            adam_eps=float(study.run.optimizer.adam_eps),
            adam_microbatch_questions=int(study.run.optimizer.adam_microbatch_questions),
            beta_lower=float(exp.beta_lower),
            beta_upper=float(exp.beta_upper),
        )
        n_steps = int(optimize.max_steps)
        step_batch_total = (
            int(adam_step_batch_total) if adam_step_batch_total is not None else 0
        )
        stale_steps = 0
        with logging_redirect_tqdm():
            for step in range(n_steps):
                with adam_step_batch_progress(
                    step=step,
                    n_steps=n_steps,
                    total=step_batch_total,
                ) as step_bar:
                    current = tuple(float(x) for x in beta_param.detach().tolist())
                    grad = grad_fn(current, eligible)
                    step_grad_norm = math.sqrt(
                        sum(float(g) * float(g) for g in grad)
                    )
                    adam.zero_grad()
                    beta_param.grad = torch.tensor(list(grad), dtype=torch.float64)
                    adam.step()
                    updated = tuple(float(x) for x in beta_param.detach().tolist())
                    for b in updated:
                        if not (exp.beta_lower <= b <= exp.beta_upper):
                            raise ValueError(f"β out of bounds after step: {updated}")
                    # DEC-084 / GRAD-006: log loss at the logged (post-step) β.
                    loss, comps = coerce_objective(objective_fn(updated, eligible))
                    metric = _metric_row(
                        index=step,
                        optimizer_kind=kind,
                        comps=comps,
                        beta=updated,
                        beta_lower=float(exp.beta_lower),
                        beta_upper=float(exp.beta_upper),
                        step_grad_norm=step_grad_norm,
                    )
                    step_rows.append(metric)
                    iteration_rows.append(
                        {k: metric[k] for k in ITERATION_CSV_COLUMNS}
                    )
                    trials.append(
                        {
                            "trial_index": step,
                            "optimizer_kind": kind,
                            "beta": list(updated),
                            "l_total": loss,
                            "question_ids": list(eligible),
                        }
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_beta = list(updated)
                        stale_steps = 0
                    else:
                        stale_steps += 1
                    best_so_far = best_loss
                    step_bar.set_postfix(
                        l_total=f"{loss:.4g}", best_l_total=f"{best_so_far:.4g}"
                    )
                    log_progress(
                        "optimize_step",
                        trial_index=step,
                        optimizer_kind=kind,
                        l_total=loss,
                        best_l_total=best_so_far,
                        n_steps=n_steps,
                        stale_steps=stale_steps,
                        patience=patience,
                    )
                    ckpt = dump_checkpoint(
                        optimizer_kind=kind,
                        beta=list(updated),
                        optimizer_state={"step": step},
                        config_hash=study_config_fingerprint(study),
                        objective_version="v1",
                        ro_manifest_hash="orch-fixed-ro",
                    )
                    (ckpt_dir / f"step_{step:04d}.json").write_text(
                        json.dumps(ckpt, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    # DEC-099: stop after ``patience`` consecutive non-improving steps.
                    if patience is not None and stale_steps >= int(patience):
                        stopped_early = True
                        break
    elif kind == "cmaes":
        from epistemic_sycophancy.optimization.cmaes import CMAESOptimizer

        if optimize.n_trials is None or optimize.population_size is None:
            raise ValueError("cmaes optimize requires n_trials and population_size")
        cma_seed = study.run.optimizer.cma_seed
        if cma_seed is None:
            raise ValueError("cmaes requires run.optimizer.cma_seed")
        from epistemic_sycophancy.prompts.ordering import build_ro_manifest

        opt = CMAESOptimizer(
            x0=beta0,
            sigma0=0.5,
            cma_seed=int(cma_seed),
            beta_lower=float(exp.beta_lower),
            beta_upper=float(exp.beta_upper),
            eligible_question_ids=eligible,
            ro_manifest=build_ro_manifest(
                ro_seed=0,
                question_ids=list(eligible),
            ),
        )
        # Approximate population via ask size; run n_trials generations.
        n_trials = int(optimize.n_trials)
        with logging_redirect_tqdm():
            trial_bar = tqdm(
                range(n_trials),
                total=n_trials,
                desc="optimize",
                unit="trial",
            )
            for trial in trial_bar:
                candidates = opt.ask()[: int(optimize.population_size)]

                def _float_objective(
                    beta: Sequence[float], qids: Sequence[str]
                ) -> float:
                    loss_f, _ = coerce_objective(objective_fn(beta, qids))
                    return loss_f

                values = [
                    opt.evaluate_candidate(
                        cand, evaluate_on_questions=_float_objective
                    )
                    for cand in candidates
                ]
                opt.tell(candidates, values)
                best_idx = min(range(len(values)), key=lambda i: values[i])
                cand = list(map(float, candidates[best_idx]))
                loss, comps = coerce_objective(objective_fn(cand, eligible))
                metric = _metric_row(
                    index=trial,
                    optimizer_kind=kind,
                    comps=comps,
                    beta=cand,
                    beta_lower=float(exp.beta_lower),
                    beta_upper=float(exp.beta_upper),
                    step_grad_norm=None,
                )
                step_rows.append(metric)
                iteration_rows.append(
                    {k: metric[k] for k in ITERATION_CSV_COLUMNS}
                )
                trials.append(
                    {
                        "trial_index": trial,
                        "optimizer_kind": kind,
                        "beta": cand,
                        "l_total": loss,
                        "question_ids": list(eligible),
                    }
                )
                best_so_far = min(best_loss, loss)
                trial_bar.set_postfix(
                    l_total=f"{loss:.4g}", best_l_total=f"{best_so_far:.4g}"
                )
                log_progress(
                    "optimize_step",
                    trial_index=trial,
                    optimizer_kind=kind,
                    l_total=loss,
                    best_l_total=best_so_far,
                    n_trials=n_trials,
                )
                if loss < best_loss:
                    best_loss = loss
                    best_beta = cand
                ckpt = dump_checkpoint(
                    optimizer_kind=kind,
                    beta=cand,
                    optimizer_state={"trial": trial},
                    config_hash=study_config_fingerprint(study),
                    objective_version="v1",
                    ro_manifest_hash=opt.ro_manifest_hash,
                )
                (ckpt_dir / f"trial_{trial:04d}.json").write_text(
                    json.dumps(ckpt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
                )
    else:
        raise ValueError(f"unsupported optimizer kind {kind!r}")

    trials_path = out_dir / "trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for row in trials:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    curve_path = plot_loss_over_trials(trials, out_dir / "loss_curve.png")

    steps_path = write_optimize_metrics_csv(
        step_rows, out_dir / "steps.csv", columns=STEP_CSV_COLUMNS
    )
    iterations_path = write_optimize_metrics_csv(
        iteration_rows, out_dir / "iterations.csv", columns=ITERATION_CSV_COLUMNS
    )
    curves_dir = out_dir / "curves"
    curve_paths = plot_iteration_metric_curves(iteration_rows, curves_dir)

    static_payload: dict[str, int] = {}
    if n_q_plus is not None:
        static_payload["n_q_plus"] = int(n_q_plus)
    if n_q_minus is not None:
        static_payload["n_q_minus"] = int(n_q_minus)
    static_path: Path | None = None
    if static_payload:
        static_path = out_dir / "static.json"
        static_path.write_text(
            json.dumps(static_payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    best_ckpt = dump_checkpoint(
        optimizer_kind=kind,
        beta=best_beta,
        optimizer_state={"best": True},
        config_hash=study_config_fingerprint(study),
        objective_version="v1",
        ro_manifest_hash="orch-fixed-ro",
    )
    best_path = out_dir / "best_checkpoint.json"
    best_path.write_text(
        json.dumps(best_ckpt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    artifacts: dict[str, str] = {
        "best_checkpoint": str(best_path),
        "trials": str(trials_path),
        "steps_csv": str(steps_path),
        "iterations_csv": str(iterations_path),
    }
    if curve_path is not None:
        artifacts["loss_curve"] = str(curve_path)
    if curve_paths:
        artifacts["curves_dir"] = str(curves_dir)
    if static_path is not None:
        artifacts["static"] = str(static_path)
    return {
        "metrics": {
            "best_beta": best_beta,
            "best_l_total": best_loss if best_loss < float("inf") else None,
            "n_trials": len(trials),
            "eligible_question_ids": list(eligible),
            "optimizer_kind": kind,
            "optimize_max_steps": optimize.max_steps,
            "patience": patience,
            "stopped_early": stopped_early,
            "n_q_plus": n_q_plus,
            "n_q_minus": n_q_minus,
        },
        "artifacts": artifacts,
    }
