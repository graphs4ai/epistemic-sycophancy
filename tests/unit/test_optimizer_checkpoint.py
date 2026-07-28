"""Optimizer checkpoint round-trip tests (Phase H OPT-009)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
def test_optimizer__checkpoint_roundtrip__preserves_beta_optimizer_state_and_config_hash() -> None:
    """OPT-009: checkpoint dump/load preserves β, optimizer state, config_hash."""
    from epistemic_sycophancy.optimization.checkpoint import (
        dump_checkpoint,
        load_checkpoint,
    )
    from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

    beta = torch.tensor([-1.0, -0.5, 0.0], dtype=torch.float64, requires_grad=True)
    optimizer = ProjectedAdam(
        beta=beta,
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=0.0,
    )
    beta.grad = torch.tensor([0.1, -0.2, 0.0], dtype=torch.float64)
    optimizer.step()

    config_hash = "a" * 64
    ro_manifest_hash = "b" * 64
    checkpoint = dump_checkpoint(
        optimizer_kind="projected_adam",
        beta=beta.detach().tolist(),
        optimizer_state=optimizer.torch_optimizer.state_dict(),
        config_hash=config_hash,
        objective_version="v1_no_residual",
        ro_manifest_hash=ro_manifest_hash,
    )
    restored = load_checkpoint(checkpoint)

    assert restored["checkpoint_version"] == "v1"
    assert restored["optimizer_kind"] == "projected_adam"
    assert restored["beta"] == beta.detach().tolist()
    assert restored["config_hash"] == config_hash
    assert restored["objective_version"] == "v1_no_residual"
    assert restored["ro_manifest_hash"] == ro_manifest_hash
    # Exact plain-state equality after round-trip (DEC-032)
    assert restored == checkpoint
    assert load_checkpoint(restored) == restored
