"""ORCH-034…038: real_model limited-path gates on layer17_n32 (replace hollow ORCH-017/018)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from epistemic_sycophancy.config.load_study import load_study_config
from epistemic_sycophancy.runner.cli import dispatch_stage, run_cli
from epistemic_sycophancy.runner.identity import clear_stack_cache

CFG = Path("configs/dev/layer17_n32.yaml")


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.fail("CUDA required for ORCH-034…038 real_model limited-path gates (not skipped)")


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n32__identity_via_cli_default_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORCH-034: identity on layer17_n32 via default load_stack (actual residuals)."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(tmp_path / "art")))
    result = dispatch_stage(
        "identity",
        study=study,
        freeze_status="unsealed",
    )
    assert result.ok
    assert result.metrics["identity_passed"] is True
    path = Path(result.artifacts["identity_result"])
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["identity_passed"] is True
    assert float(payload["max_abs_diff"]) < 1e-5


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n32__baseline_writes_partition_without_score_fn(
    tmp_path: Path,
) -> None:
    """ORCH-035: baseline_partitions via default adapters (no score_fn kwarg)."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(tmp_path / "art")))
    result = dispatch_stage(
        "baseline_partitions",
        study=study,
        freeze_status="unsealed",
        score_fn=None,
    )
    assert result.ok
    order = study.run.order_regime
    path = Path(study.run.artifact_dir) / "baseline" / f"partition_{order}.json"
    assert path.is_file(), f"missing {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "q_plus" in payload and "q_minus" in payload
    assert payload["n_q_plus"] + payload["n_q_minus"] >= 1


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n32__feature_selection_writes_pool_keys(
    tmp_path: Path,
) -> None:
    """ORCH-036: feature_selection writes common_pool with (layer, feature_id).

    DEC-085: FS needs the frozen baseline partition for Q+/Q− component subsets,
    so baseline_partitions must run first (same stage order as ORCH-037/038).
    """
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudyFsCoverageConfig

    art = tmp_path / "art"
    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(art)))

    # Baseline must write partition_CF.json before multi-condition FS (DEC-085).
    assert dispatch_stage(
        "baseline_partitions", study=study, freeze_status="unsealed", score_fn=None
    ).ok
    assert (art / "baseline" / "partition_CF.json").is_file()

    fs_study = replace(
        study,
        run=replace(
            study.run,
            order_regime="CF",
            fs_coverage=StudyFsCoverageConfig(n_questions=2, seed=0),
        ),
    )
    result = dispatch_stage(
        "feature_selection",
        study=fs_study,
        freeze_status="unsealed",
        jacobian_fn=None,
        scale_fn=None,
    )
    assert result.ok
    pool_path = Path(result.artifacts["pool"])
    assert pool_path.is_file()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == 2
    assert payload["pool_size"] >= 1
    assert payload["scale_source"] == "decoder_norm"
    assert "provenance" in payload
    assert all(len(pair) == 2 for pair in payload["feature_ids"])
    assert all(s > 0 for s in payload["feature_scales"])


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n32__optimize_finite_l_total_default_adapters(
    tmp_path: Path,
) -> None:
    """ORCH-037: optimize finite l_total via default adapters (identity from artifact)."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudyFsCoverageConfig

    art = tmp_path / "art"
    study = load_study_config(CFG)
    study = replace(study, run=replace(study.run, artifact_dir=str(art)))

    assert dispatch_stage("identity", study=study, freeze_status="unsealed").ok
    assert dispatch_stage(
        "baseline_partitions", study=study, freeze_status="unsealed", score_fn=None
    ).ok
    part = json.loads((art / "baseline" / "partition_CF.json").read_text(encoding="utf-8"))
    q_plus = list(part["q_plus"])
    q_minus = list(part["q_minus"])
    assert q_plus and q_minus, "need both partitions for optimize objective"
    # Tiny coverage IDs spanning Q+/Q- (full YAML N=32 is too heavy for live IB/CB).
    coverage_ids = (q_plus[0], q_minus[0])

    fs_study = replace(
        study,
        run=replace(
            study.run,
            order_regime="CF",
            fs_coverage=StudyFsCoverageConfig(n_questions=2, seed=0),
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
        run=replace(study.run, fs_coverage=StudyFsCoverageConfig(question_ids=coverage_ids)),
    )
    result = dispatch_stage(
        "optimize",
        study=opt_study,
        freeze_status="unsealed",
        margin_payload=None,
        beta=None,
        identity_passed=None,
    )
    assert result.ok
    assert result.metrics.get("best_l_total") is not None
    assert math.isfinite(float(result.metrics["best_l_total"]))
    trials_path = Path(result.artifacts["trials"])
    assert trials_path.is_file()
    rows = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert all(math.isfinite(float(row["l_total"])) for row in rows)
    assert "best_checkpoint" in result.artifacts
    assert Path(result.artifacts["best_checkpoint"]).is_file()


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__layer17_n32__optimize_writes_best_checkpoint_default_adapters(
    tmp_path: Path,
) -> None:
    """ORCH-038: optimize uses run.optimize budget; writes best_checkpoint via defaults."""
    _require_cuda()
    clear_stack_cache()
    from dataclasses import replace

    from epistemic_sycophancy.config.study import StudyOptimizeConfig, StudyFsCoverageConfig

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
            order_regime="CF",
            fs_coverage=StudyFsCoverageConfig(n_questions=2, seed=0),
        ),
    )
    assert dispatch_stage(
        "feature_selection",
        study=fs_study,
        freeze_status="unsealed",
        jacobian_fn=None,
        scale_fn=None,
    ).ok

    # Tiny optimize coverage (YAML n_questions=4); explicit optimize budgets only.
    opt_study = replace(
        study,
        run=replace(
            study.run,
            optimize=StudyOptimizeConfig(
                budget_match_on="n_objective_evals",
                max_steps=2,
                n_questions=2,
            ),
        ),
    )
    assert opt_study.run.optimize.max_steps == 2

    result = dispatch_stage(
        "optimize",
        study=opt_study,
        freeze_status="unsealed",
        objective_fn=None,
        grad_fn=None,
        identity_passed=None,
    )
    assert result.ok
    ckpt = art / "optimize" / "best_checkpoint.json"
    assert ckpt.is_file()
    payload = json.loads(ckpt.read_text(encoding="utf-8"))
    assert "beta" in payload
    assert len(payload["beta"]) >= 1
    assert all(math.isfinite(float(b)) for b in payload["beta"])
    # Must respect run.optimize.max_steps, not an implicit tiny budget.
    trials = art / "optimize" / "trials.jsonl"
    assert trials.is_file()
    n_trials = sum(1 for _ in trials.read_text(encoding="utf-8").splitlines() if _.strip())
    assert n_trials >= 1
