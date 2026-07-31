"""ORCH-007: CLI --config end-to-end with fake stack_loader."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest
import torch
import yaml

from epistemic_sycophancy.config.study import StudyConfig


class _FakeStack:
    def __init__(self, study: StudyConfig) -> None:
        self.config = study.stack
        self._residual = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)

    def capture_layer_residuals(self, *, texts: Sequence[str], layers: Sequence[int]):
        del texts
        return {int(layer): self._residual.clone() for layer in layers}

    @contextmanager
    def install_hooks(self, **kwargs: Any) -> Iterator[None]:
        del kwargs
        yield


@pytest.mark.unit
def test_cli_main__config_identity_through_optimize__with_fake_stack_loader(
    tmp_path: Path,
) -> None:
    """ORCH-007: run_cli with stack_loader runs identity without monkeypatching dispatch."""
    from epistemic_sycophancy.runner.cli import run_cli
    from epistemic_sycophancy.runner.identity import clear_stack_cache

    clear_stack_cache()
    # Minimal valid Study YAML (same schema as limited dev preset).
    cfg_path = tmp_path / "study.yaml"
    payload = yaml.safe_load(
        Path("configs/dev/layer17_n32.yaml").read_text(encoding="utf-8")
    )
    payload["run"]["artifact_dir"] = str(tmp_path / "artifacts")
    payload["run"]["fs_coverage"] = {"question_ids": ["q1", "q2"]}
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loads: list[StudyConfig] = []

    def stack_loader(study: StudyConfig) -> _FakeStack:
        loads.append(study)
        return _FakeStack(study)

    code = run_cli(
        ["identity", "--config", str(cfg_path)],
        stack_loader=stack_loader,
    )
    assert code == 0
    assert loads, "run_cli must pass stack_loader into dispatch (DEC-065)"
