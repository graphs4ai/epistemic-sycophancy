"""Resolve identity_passed across CLI processes (ORCH-026 / DEC-074)."""

from __future__ import annotations

import json
from pathlib import Path

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.reproducibility.phase_gates import require_identity_gate


def resolve_identity_passed(study: StudyConfig) -> bool:
    """Read ``{artifact_dir}/identity/identity_result.json`` (DEC-074).

    Missing artifact → ``ValueError``. ``identity_passed=false`` →
    ``OptimizationBlockedError``. Never defaults to True.
    """
    path = Path(study.run.artifact_dir) / "identity" / "identity_result.json"
    if not path.is_file():
        raise ValueError(
            f"identity_result artifact missing at {path}; run identity stage first "
            "(ORCH-026 / DEC-074)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "identity_passed" not in payload:
        raise ValueError(f"identity_result missing identity_passed field: {path}")
    passed = bool(payload["identity_passed"])
    require_identity_gate(identity_passed=passed)
    return True
