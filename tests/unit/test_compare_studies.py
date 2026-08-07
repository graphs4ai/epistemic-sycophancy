"""Unit test for suppression vs bidirectional validation comparison helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.unit
def test_compare_studies__reads_behavioral_and_diagnostics(
    tmp_path: Path,
) -> None:
    """ORCH-DIAG-003: side-by-side summary loader for two full_study dirs."""
    from epistemic_sycophancy.analysis.compare_studies import compare_validation_dirs

    for name, ftw in (("sup", 0.3), ("bid", 0.2)):
        d = tmp_path / name
        d.mkdir()
        (d / "behavioral.json").write_text(
            json.dumps({"ftw": ftw, "cbr": 0.7, "selectivity": 0.4}),
            encoding="utf-8",
        )
        (d / "margin_subset_summary_best_by_l_total.json").write_text(
            json.dumps({"resistance": {"baseline_failing": {"n": 1}}}),
            encoding="utf-8",
        )
        (d / "context_contrast_summary_best_by_l_total.json").write_text(
            json.dumps({"q_plus": {"mean_delta_d_r": 0.1}}),
            encoding="utf-8",
        )

    result = compare_validation_dirs(
        suppression_dir=tmp_path / "sup",
        bidirectional_dir=tmp_path / "bid",
    )
    assert result["suppression"]["behavioral"]["ftw"] == 0.3
    assert result["bidirectional"]["behavioral"]["ftw"] == 0.2
    assert result["suppression"]["margin_subsets"]["resistance"]["baseline_failing"]["n"] == 1
    assert result["bidirectional"]["context_contrast"]["q_plus"]["mean_delta_d_r"] == 0.1
