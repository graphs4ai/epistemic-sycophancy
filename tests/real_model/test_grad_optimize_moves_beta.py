"""GRAD-008: layer17_n2 optimize must move β (or loud-fail), never silent flat zeros."""

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
def test_real_model__layer17_n2__optimize_moves_beta_or_loud_fails(
    tmp_path: Path,
) -> None:
    """GRAD-008: after fix, trials must not be all-zero β for every step.

    Accepts either (a) at least one |β_i| > 0 in any trial, or (b) a loud
    DEC-084 identically-zero grad failure. Silent flat all-zero trials fail.
    """
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudyOptimizeConfig, StudySmokeConfig

    art = tmp_path / "art"
    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(art)))

    assert dispatch_stage("identity", study=study, freeze_status="unsealed").ok
    assert dispatch_stage(
        "baseline_partitions", study=study, freeze_status="unsealed", score_fn=None
    ).ok
    fs_study = replace(
        study,
        run=replace(
            study.run,
            order_regimes=("CF",),
            smoke=StudySmokeConfig(n_questions=2, split="feature_selection", seed=0),
        ),
    )
    assert dispatch_stage(
        "feature_selection",
        study=fs_study,
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
            return  # loud degenerate-grad diagnosis (DEC-084)
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
        "GRAD-008 failure: all trials have all-zero β "
        "(obsolete flat artifact is not success evidence)"
    )
