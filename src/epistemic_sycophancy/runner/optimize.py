"""Non-smoke optimize stage (ORCH-009 / DEC-066 / DEC-072)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.logging.loss_curve import plot_loss_over_trials
from epistemic_sycophancy.optimization.checkpoint import dump_checkpoint
from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam
from epistemic_sycophancy.reproducibility.phase_gates import require_identity_gate


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
    objective_fn: Callable[[Sequence[float], Sequence[str]], float],
    grad_fn: Callable[[Sequence[float], Sequence[str]], Sequence[float]] | None = None,
    beta_init: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run non-smoke optimize using ``run.optimize`` budgets (never smoke max_steps)."""
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
    best_beta = list(beta0)
    best_loss = float("inf")

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
        for step in range(int(optimize.max_steps)):
            current = tuple(float(x) for x in beta_param.detach().tolist())
            grad = grad_fn(current, eligible)
            adam.zero_grad()
            beta_param.grad = torch.tensor(list(grad), dtype=torch.float64)
            adam.step()
            updated = tuple(float(x) for x in beta_param.detach().tolist())
            for b in updated:
                if not (exp.beta_lower <= b <= exp.beta_upper):
                    raise ValueError(f"β out of bounds after step: {updated}")
            # DEC-084 / GRAD-006: log loss at the logged (post-step) β.
            loss = float(objective_fn(updated, eligible))
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
            ckpt = dump_checkpoint(
                optimizer_kind=kind,
                beta=list(updated),
                optimizer_state={"step": step},
                config_hash=study_config_fingerprint(study),
                objective_version="v1",
                ro_manifest_hash="orch-fixed-ro",
            )
            (ckpt_dir / f"step_{step:04d}.json").write_text(
                json.dumps(ckpt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
    elif kind == "cmaes":
        from epistemic_sycophancy.optimization.cmaes import CMAESOptimizer

        if optimize.n_trials is None or optimize.population_size is None:
            raise ValueError("cmaes optimize requires n_trials and population_size")
        cma_seed = study.run.optimizer.cma_seed
        if cma_seed is None:
            raise ValueError("cmaes requires run.optimizer.cma_seed")
        opt = CMAESOptimizer(
            x0=beta0,
            sigma0=0.5,
            cma_seed=int(cma_seed),
            beta_lower=float(exp.beta_lower),
            beta_upper=float(exp.beta_upper),
            eligible_question_ids=eligible,
            ro_manifest={"order": "RO", "seed": 0},
        )
        # Approximate population via ask size; run n_trials generations.
        for trial in range(int(optimize.n_trials)):
            candidates = opt.ask()[: int(optimize.population_size)]
            values = [
                opt.evaluate_candidate(cand, evaluate_on_questions=objective_fn)
                for cand in candidates
            ]
            opt.tell(candidates, values)
            best_idx = min(range(len(values)), key=lambda i: values[i])
            cand = list(map(float, candidates[best_idx]))
            loss = float(values[best_idx])
            trials.append(
                {
                    "trial_index": trial,
                    "optimizer_kind": kind,
                    "beta": cand,
                    "l_total": loss,
                    "question_ids": list(eligible),
                }
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
    }
    if curve_path is not None:
        artifacts["loss_curve"] = str(curve_path)
    return {
        "metrics": {
            "best_beta": best_beta,
            "best_l_total": best_loss if best_loss < float("inf") else None,
            "n_trials": len(trials),
            "eligible_question_ids": list(eligible),
            "optimizer_kind": kind,
            "used_smoke_max_steps": False,
            "optimize_max_steps": optimize.max_steps,
        },
        "artifacts": artifacts,
    }
