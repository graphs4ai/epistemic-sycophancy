"""Compare suppression-only vs bidirectional full_study validation summaries.

Usage:
  pixi run python -m epistemic_sycophancy.analysis.compare_studies \\
    --suppression artifacts/.../full_study \\
    --bidirectional artifacts/.../full_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_validation_dirs(
    *,
    suppression_dir: Path,
    bidirectional_dir: Path,
) -> dict[str, Any]:
    """Side-by-side behavioral + subset + contrast summaries for l_total."""
    out: dict[str, Any] = {"suppression": {}, "bidirectional": {}}
    for label, directory in (
        ("suppression", suppression_dir),
        ("bidirectional", bidirectional_dir),
    ):
        behavioral = directory / "behavioral.json"
        subset = directory / "margin_subset_summary_best_by_l_total.json"
        contrast = directory / "context_contrast_summary_best_by_l_total.json"
        entry: dict[str, Any] = {}
        if behavioral.is_file():
            payload = _load_json(behavioral)
            entry["behavioral"] = {
                k: payload.get(k)
                for k in (
                    "neutral_accuracy",
                    "ftw",
                    "cbr",
                    "selectivity",
                    "pra_mean",
                    "pra_all",
                )
            }
        if subset.is_file():
            entry["margin_subsets"] = _load_json(subset)
        if contrast.is_file():
            entry["context_contrast"] = _load_json(contrast)
        out[label] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suppression", type=Path, required=True)
    parser.add_argument("--bidirectional", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write comparison JSON",
    )
    args = parser.parse_args(argv)
    result = compare_validation_dirs(
        suppression_dir=args.suppression,
        bidirectional_dir=args.bidirectional,
    )
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
