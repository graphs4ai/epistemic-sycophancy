"""GRAD-008 / GRAD-011: layer17_n2 optimize must move β; never silent flat zeros."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.runner.cli import dispatch_stage
from epistemic_sycophancy.runner.identity import clear_stack_cache

CFG = Path("configs/smokes/layer17_n2.yaml")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GRAD-008 real_model optimize")


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n2__optimize_moves_beta(
    tmp_path: Path,
) -> None:
    """GRAD-011: assert β moves. DEC-084 identically-zero grad is failure, not success.

    Silent flat all-zero trials fail. Loud zero-grad is also failure (record as
    blocked with FS activity diagnosis until FSC-009); do not swallow it.
    """
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudyOptimizeConfig

    art = tmp_path / "art"
    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(art)))

    assert dispatch_stage("identity", study=study, freeze_status="unsealed").ok
    assert dispatch_stage(
        "baseline_partitions", study=study, freeze_status="unsealed", score_fn=None
    ).ok
    # Use config smoke N (32); do not shrink to N=2 (DEC-079 / GRAD-011).
    assert dispatch_stage(
        "feature_selection",
        study=replace(study, run=replace(study.run, order_regime="CF")),
        freeze_status="unsealed",
        jacobian_fn=None,
        scale_fn=None,
    ).ok

    opt_study = replace(
        study,
        run=replace(
            study.run,
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=5,
                n_questions=2,
            ),
        ),
    )

    try:
        result = dispatch_stage(
            "optimize",
            study=opt_study,
            freeze_status="unsealed",
            objective_fn=None,
            grad_fn=None,
            identity_passed=None,
        )
    except ValueError as exc:
        msg = str(exc)
        if "identically zero" in msg or "non-finite" in msg or "∂L/∂β" in msg:
            pytest.fail(
                "GRAD-011: DEC-084 identically-zero ∂L/∂β is not success. "
                "Likely N-only FS pool (features inactive on IB/CB); "
                "see DEC-085 / FSC-009. "
                f"Original: {msg}"
            )
        raise

    assert result.ok
    trials_path = art / "optimize" / "trials.jsonl"
    assert trials_path.is_file()
    rows = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "expected nonempty trials.jsonl"
    moved = False
    for row in rows:
        beta = [float(b) for b in row["beta"]]
        assert all(math.isfinite(b) for b in beta)
        if any(abs(b) > 1e-8 for b in beta):
            moved = True
            break
    assert moved, (
        "GRAD-011 failure: all trials have all-zero β "
        "(obsolete flat artifact is not success evidence)"
    )
