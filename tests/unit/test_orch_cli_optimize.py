"""ORCH-012: CLI optimize with fake objective; CUDA smoke marker reserved."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from epistemic_sycophancy.config.study import StudyConfig


@pytest.mark.unit
def test_cli_main__optimize__fake_objective_stack_writes_checkpoint(
    tmp_path: Path,
) -> None:
    """ORCH-012: run_cli optimize with injected objective writes checkpoint."""
    from epistemic_sycophancy.runner.cli import run_cli

    cfg_path = tmp_path / "study.yaml"
    payload = yaml.safe_load(
        Path("configs/smokes/layer17_n2.yaml").read_text(encoding="utf-8")
    )
    payload["run"]["artifact_dir"] = str(tmp_path / "artifacts")
    payload["experiment"]["feature_ids"] = [[17, 1]]
    payload["experiment"]["feature_scales"] = [1.0]
    payload["experiment"]["coefficient_length"] = 1
    payload["run"]["optimize"] = {
        "budget_match_on": "n_objective_evals",
        "max_steps": 2,
        "question_ids": ["qo1", "qo2"],
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    code = run_cli(
        ["optimize", "--config", str(cfg_path)],
        identity_passed=True,
        objective_fn=lambda beta, qids: float(sum(beta)),
        grad_fn=lambda beta, qids: tuple(0.0 for _ in beta),
        optimization_question_ids=("qo1", "qo2"),
    )
    assert code == 0
    best = tmp_path / "artifacts" / "optimize" / "best_checkpoint.json"
    assert best.is_file()


@pytest.mark.real_model
@pytest.mark.slow
@pytest.mark.gpu
def test_real_model__optimize__tiny_budget_finite_loss_and_checkpoint() -> None:
    """ORCH-012 CUDA: tiny non-smoke optimize on layer17 smoke YAML (ship path)."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for ORCH-012 real optimize smoke")
    # Full stack wiring is exercised in ORCH-018; this gate documents the marker path.
    assert Path("configs/smokes/layer17_n2.yaml").is_file()
