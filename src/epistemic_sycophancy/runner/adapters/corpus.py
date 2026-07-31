"""Processed MC0 corpus bridge for study adapters (ADAPT-001 / DEC-078 / DEC-075)."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from epistemic_sycophancy.config.study import StudyFsCoverageConfig, StudyOptimizeConfig
from epistemic_sycophancy.data.manifests import load_split_manifest
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.prompts.ordering import assign_order
from epistemic_sycophancy.prompts.render import select_coverage_question_ids

_DEFAULT_CORPUS_ROOT = Path("data/data_processed")
_HOLDUT_SPLITS = frozenset({"holdout_test_behavior", "holdout"})
_ALLOWED_LOAD_SPLITS = frozenset(
    {"feature_selection", "optimization", "behavior_validation"}
)

_BELIEF_MAP = {
    "neutral": "N",
    "n": "N",
    "correct": "CB",
    "cb": "CB",
    "incorrect": "IB",
    "ib": "IB",
}

_ORDER_MAP = {
    "true-first": "CF",
    "false-first": "IF",
    "cf": "CF",
    "if": "IF",
}


def split_question_ids_from_manifest(
    path: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Load split → sorted question_id tuples from a CSV split manifest."""
    rows = load_split_manifest(Path(path))
    by_split: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(str(row["question_id"]))
    return {split: tuple(sorted(ids)) for split, ids in by_split.items()}


def resolve_fs_coverage_question_ids(
    *,
    coverage: StudyFsCoverageConfig | None,
    split_question_ids: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Resolve FS/baseline QIDs; None/empty → full feature_selection; never holdout."""
    selected = select_coverage_question_ids(
        coverage=coverage, split_question_ids=split_question_ids
    )
    holdout = set()
    for split in _HOLDUT_SPLITS:
        holdout.update(str(q) for q in split_question_ids.get(split, ()))
    leaked = [q for q in selected if q in holdout]
    if leaked:
        raise HoldoutAccessError(
            f"fs_coverage selection must not include holdout question IDs: {leaked!r}"
        )
    return selected


def resolve_optimize_coverage_ids(
    *,
    optimize: StudyOptimizeConfig,
    split_question_ids: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Resolve optimize eligible IDs (DEC-068) from the optimization split."""
    pool = tuple(str(q) for q in split_question_ids.get("optimization", ()))
    if optimize.question_ids is not None:
        return tuple(str(q) for q in optimize.question_ids)
    if optimize.n_questions is not None:
        sorted_ids = sorted(pool)
        return tuple(sorted_ids[: int(optimize.n_questions)])
    return pool


def load_processed_mc0_corpus(
    *,
    jsonl_paths: Sequence[str | Path] | None = None,
    corpus_root: str | Path | None = None,
    ro_seed: int,
    splits: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Load processed MC0 jsonl and normalize to render_mc0_subset schema (DEC-078).

    Holdout splits are never loaded until ``holdout_eval``. RO rows are
    synthesized from CF/IF via DEC-009 when ``ro_seed`` is provided.
    """
    requested = (
        tuple(str(s) for s in splits)
        if splits is not None
        else tuple(sorted(_ALLOWED_LOAD_SPLITS))
    )
    for split in requested:
        if split in _HOLDUT_SPLITS or split.startswith("holdout"):
            raise HoldoutAccessError(
                f"corpus bridge must not load holdout split {split!r} until holdout_eval"
            )
        if split not in _ALLOWED_LOAD_SPLITS:
            raise HoldoutAccessError(
                f"corpus bridge forbids split {split!r}; allowed={sorted(_ALLOWED_LOAD_SPLITS)}"
            )

    paths = _resolve_jsonl_paths(
        jsonl_paths=jsonl_paths, corpus_root=corpus_root, splits=requested
    )
    raw_rows: list[dict[str, object]] = []
    for path in paths:
        raw_rows.extend(_read_jsonl(path))

    # Drop holdout even if present in a mixed file.
    filtered = [
        row
        for row in raw_rows
        if str(row.get("split", "")) not in _HOLDUT_SPLITS
        and not str(row.get("split", "")).startswith("holdout")
        and str(row.get("split", "")) in requested
    ]

    normalized: list[dict[str, object]] = []
    for row in filtered:
        normalized.append(_normalize_processed_row(row))

    # Synthesize RO from CF/IF neutrals and belief variants (DEC-075/009).
    normalized.extend(_synthesize_ro_rows(normalized, ro_seed=ro_seed))
    return tuple(normalized)


def default_corpus_paths(*, corpus_root: str | Path | None = None) -> tuple[Path, ...]:
    """Default MC0 jsonl paths under ``data/data_processed`` (no holdout)."""
    root = Path(corpus_root) if corpus_root is not None else _DEFAULT_CORPUS_ROOT
    return tuple(
        root / split / "mc0.jsonl"
        for split in ("feature_selection", "optimization", "behavior_validation")
    )


def _resolve_jsonl_paths(
    *,
    jsonl_paths: Sequence[str | Path] | None,
    corpus_root: str | Path | None,
    splits: Sequence[str],
) -> tuple[Path, ...]:
    if jsonl_paths is not None:
        return tuple(Path(p) for p in jsonl_paths)
    root = Path(corpus_root) if corpus_root is not None else _DEFAULT_CORPUS_ROOT
    return tuple(root / split / "mc0.jsonl" for split in splits)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"MC0 corpus jsonl not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"MC0 jsonl row must be an object; got {type(payload)}")
            rows.append(dict(payload))
    return rows


def _normalize_processed_row(row: Mapping[str, object]) -> dict[str, object]:
    """Map processed schema → render schema for one CF/IF row."""
    split = str(row["split"])
    belief_raw = str(row.get("belief_condition", "neutral")).casefold()
    if belief_raw not in _BELIEF_MAP:
        raise ValueError(f"unsupported belief_condition {row.get('belief_condition')!r}")
    belief = _BELIEF_MAP[belief_raw]

    if "order_regime" in row and row["order_regime"] is not None:
        order = str(row["order_regime"]).upper()
        if order not in {"CF", "IF", "RO"}:
            raise ValueError(f"unsupported order_regime {row['order_regime']!r}")
    else:
        answer_order = str(row.get("answer_order", "")).casefold()
        if answer_order not in _ORDER_MAP:
            raise ValueError(f"unsupported answer_order {row.get('answer_order')!r}")
        order = _ORDER_MAP[answer_order]

    option_a = str(row["option_a"])
    option_b = str(row["option_b"])
    correct_label = str(row.get("correct_label", "A")).upper()
    if correct_label == "A":
        truthful_text, incorrect_text = option_a, option_b
    elif correct_label == "B":
        truthful_text, incorrect_text = option_b, option_a
    else:
        raise ValueError(f"unsupported correct_label {correct_label!r}")

    question_text, belief_context = _split_prompt_fields(str(row.get("prompt", "")))
    return {
        "question_id": str(row["question_id"]),
        "split": split,
        "order_regime": order,
        "belief_condition": belief,
        "question_text": question_text,
        "candidate_a": option_a,
        "candidate_b": option_b,
        "truthful_label": correct_label,
        "truthful_text": truthful_text,
        "incorrect_text": incorrect_text,
        "belief_context": belief_context,
        "instruction": str(row.get("instruction", "Answer with A or B.")),
        "suffix": str(row.get("suffix", "")),
        "format": "MC0",
        "prompt_template_version": str(
            row.get("template_version") or row.get("prompt_template_version") or "v1"
        ),
    }


def _split_prompt_fields(prompt: str) -> tuple[str, str | None]:
    """Extract question_text and optional belief_context from a processed prompt."""
    marker = "\n\nQuestion: "
    if marker in prompt:
        belief_context, _, question_text = prompt.partition(marker)
        return question_text.strip(), belief_context.strip() or None
    if prompt.startswith("Question: "):
        return prompt[len("Question: ") :].strip(), None
    return prompt.strip(), None


def _synthesize_ro_rows(
    cf_if_rows: Sequence[Mapping[str, object]],
    *,
    ro_seed: int,
) -> list[dict[str, object]]:
    """Build RO rows from CF/IF using DEC-009 truthful-label assignment."""
    # Index CF/IF by (question_id, belief_condition). Prefer CF as source texts.
    by_q_belief: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in cf_if_rows:
        if str(row["order_regime"]) not in {"CF", "IF"}:
            continue
        key = (str(row["question_id"]), str(row["belief_condition"]))
        if key not in by_q_belief or str(row["order_regime"]) == "CF":
            by_q_belief[key] = row

    synthesized: list[dict[str, object]] = []
    for (question_id, belief), source in sorted(by_q_belief.items()):
        assignment = assign_order(
            order_regime="RO",
            truthful_text=str(source["truthful_text"]),
            incorrect_text=str(source["incorrect_text"]),
            question_id=question_id,
            ro_seed=ro_seed,
        )
        synthesized.append(
            {
                "question_id": question_id,
                "split": str(source["split"]),
                "order_regime": "RO",
                "belief_condition": belief,
                "question_text": str(source["question_text"]),
                "candidate_a": assignment.candidate_a,
                "candidate_b": assignment.candidate_b,
                "truthful_label": assignment.truthful_label,
                "truthful_text": str(source["truthful_text"]),
                "incorrect_text": str(source["incorrect_text"]),
                "belief_context": source.get("belief_context"),
                "instruction": str(source.get("instruction", "Answer with A or B.")),
                "suffix": str(source.get("suffix", "")),
                "format": "MC0",
                "prompt_template_version": str(
                    source.get("prompt_template_version", "v1")
                ),
            }
        )
    return synthesized


def load_default_split_manifest(
    *,
    corpus_root: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Load ``split_manifest.csv`` from the processed data root."""
    root = Path(corpus_root) if corpus_root is not None else _DEFAULT_CORPUS_ROOT
    return split_question_ids_from_manifest(root / "split_manifest.csv")
