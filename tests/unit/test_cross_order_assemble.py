"""ORDER-EXP-002: cross_order_assemble loads three sealed single-order βs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_sycophancy.runner.cross_order_assemble import (
    CrossOrderCampaignConfig,
    run_cross_order_assemble,
)


def _seed_sealed_study(root: Path, *, order: str, beta: list[float]) -> None:
    (root / "freeze").mkdir(parents=True)
    (root / "optimize").mkdir(parents=True)
    (root / "baseline").mkdir(parents=True)
    (root / "freeze" / "frozen_experiment_config.json").write_text(
        json.dumps(
            {
                "freeze_status": "sealed",
                "order_regime": order,
                "run": {"order_regime": order},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "optimize" / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint_version": "v1",
                "optimizer_kind": "projected_adam",
                "beta": beta,
                "optimizer_state": {},
                "config_hash": f"cfg-{order}",
                "objective_version": "v1",
                "ro_manifest_hash": "ro",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "baseline" / f"partition_{order}.json").write_text(
        json.dumps(
            {
                "order_regime": order,
                "fingerprint": f"fp-{order}",
                "q_plus": ["q1"],
                "q_minus": ["q2"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_cross_order_assemble__three_sealed_studies__nine_cells_distinct_betas(
    tmp_path: Path,
) -> None:
    """ORDER-EXP-002: assemble uses three distinct βs; never clones one checkpoint."""
    cf = tmp_path / "cf"
    iff = tmp_path / "if"
    ro = tmp_path / "ro"
    _seed_sealed_study(cf, order="CF", beta=[-0.1])
    _seed_sealed_study(iff, order="IF", beta=[-0.2])
    _seed_sealed_study(ro, order="RO", beta=[-0.3])

    out = tmp_path / "assemble"
    campaign = CrossOrderCampaignConfig(
        sources={"CF": str(cf), "IF": str(iff), "RO": str(ro)},
        artifact_dir=str(out),
    )
    metrics = {
        order: {
            "ftw": 0.5,
            "cbr": 0.5,
            "selectivity": 0.0,
            "n_q_plus": 1,
            "n_q_minus": 1,
        }
        for order in ("CF", "IF", "RO")
    }
    result = run_cross_order_assemble(
        campaign=campaign,
        metrics_by_evaluated_under=metrics,
    )
    assert result["metrics"]["n_cells"] == 9
    matrix = json.loads(Path(result["artifacts"]["cross_order_matrix"]).read_text())
    cells = matrix["cells"]
    assert len(cells) == 9
    by_opt = {c["optimized_under"]: tuple(c["beta"]) for c in cells}
    assert by_opt["CF"] == (-0.1,)
    assert by_opt["IF"] == (-0.2,)
    assert by_opt["RO"] == (-0.3,)
    sources = json.loads(Path(result["artifacts"]["sources"]).read_text())
    assert sources["sources"]["CF"]["beta"] == [-0.1]


@pytest.mark.unit
def test_cross_order_assemble__order_mismatch__raises(tmp_path: Path) -> None:
    """ORDER-EXP-002: source key must match sealed order_regime."""
    cf = tmp_path / "cf"
    iff = tmp_path / "if"
    ro = tmp_path / "ro"
    _seed_sealed_study(cf, order="IF", beta=[-0.1])  # mismatched
    _seed_sealed_study(iff, order="IF", beta=[-0.2])
    _seed_sealed_study(ro, order="RO", beta=[-0.3])
    campaign = CrossOrderCampaignConfig(
        sources={"CF": str(cf), "IF": str(iff), "RO": str(ro)},
        artifact_dir=str(tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="order_regime"):
        run_cross_order_assemble(
            campaign=campaign,
            metrics_by_evaluated_under={
                o: {
                    "ftw": 0.0,
                    "cbr": 0.0,
                    "selectivity": 0.0,
                    "n_q_plus": 1,
                    "n_q_minus": 1,
                }
                for o in ("CF", "IF", "RO")
            },
        )
