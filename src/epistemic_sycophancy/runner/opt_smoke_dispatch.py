"""Opt-smoke orchestration (ORCH-005 / DEC-062)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.load_study import study_config_fingerprint
from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.logging.pipeline import log_progress
from epistemic_sycophancy.runner.opt_smoke import run_opt_smoke, run_opt_smoke_adam_step


def run_opt_smoke_dispatch(
    *,
    study: StudyConfig,
    freeze_status: str,
    identity_passed: bool,
    margin_payload: Mapping[str, Any],
    beta: Sequence[float],
    adam_grad: Sequence[float] | None = None,
    question_ids: Sequence[str] | None = None,
    split_name: str = "optimization",
) -> dict[str, Any]:
    """Evaluate finite objective on smoke margins; optional ProjectedAdam step."""
    smoke = study.run.smoke
    if question_ids is not None:
        qids = tuple(str(q) for q in question_ids)
    elif smoke.question_ids is not None:
        qids = tuple(smoke.question_ids)
    else:
        qids = tuple(str(q) for q in margin_payload.get("q_plus", ())) + tuple(
            str(q) for q in margin_payload.get("q_minus", ())
        )

    exp = study.experiment
    opt = study.run.optimizer
    smoke_result = run_opt_smoke(
        question_ids=qids,
        split_name=split_name,
        beta=beta,
        freeze_status=freeze_status,
        identity_passed=identity_passed,
        tau=float(exp.tau),
        w_r=float(exp.w_r),
        w_u=float(exp.w_u),
        delta_n=float(exp.delta_n),
        delta_c=float(exp.delta_c),
        lambda_n=float(exp.lambda_n),
        lambda_c=float(exp.lambda_c),
        lambda_beta=float(exp.lambda_beta),
        ib_margins_by_question=margin_payload["ib_margins_by_question"],
        cb_margins_by_question=margin_payload["cb_margins_by_question"],
        baseline_cb_margins=margin_payload["baseline_cb_margins"],
        baseline_neutral_margins=margin_payload["baseline_neutral_margins"],
        current_neutral_margins=margin_payload["current_neutral_margins"],
        q_plus=margin_payload["q_plus"],
        q_minus=margin_payload["q_minus"],
    )
    l_total = float(smoke_result.l_total)
    if not math.isfinite(l_total):
        raise ValueError(f"opt_smoke l_total must be finite; got {l_total!r}")

    beta_after = tuple(float(b) for b in beta)
    if (
        adam_grad is not None
        and opt.kind == "projected_adam"
        and opt.max_steps >= 1
    ):
        beta_after = run_opt_smoke_adam_step(
            beta_init=beta,
            grad=adam_grad,
            adam_lr=float(opt.adam_lr),
            adam_beta1=float(opt.adam_beta1),
            adam_beta2=float(opt.adam_beta2),
            adam_eps=float(opt.adam_eps),
            adam_microbatch_questions=int(opt.adam_microbatch_questions),
            beta_lower=float(exp.beta_lower),
            beta_upper=float(exp.beta_upper),
            max_steps=int(opt.max_steps),
        )

    fingerprint = study_config_fingerprint(study)
    out_dir = Path(study.run.artifact_dir) / "opt_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "opt_smoke_result.json"
    payload = {
        "l_total": l_total,
        "beta": list(beta),
        "beta_after": list(beta_after),
        "question_ids": list(qids),
        "split_name": smoke_result.split_name,
        "holdout_accessed": smoke_result.holdout_accessed,
        "study_yaml_fingerprint": fingerprint,
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    log_progress(
        "opt_smoke",
        l_total=l_total,
        n_questions=len(qids),
        path=str(path),
    )
    return {
        "metrics": {
            "l_total": l_total,
            "beta_after": list(beta_after),
            "holdout_accessed": False,
        },
        "artifacts": {"opt_smoke": str(path)},
    }
