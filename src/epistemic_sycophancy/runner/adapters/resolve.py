"""Shared corpus resolution for dispatch defaults (ORCH-027+)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.runner.adapters.corpus import (
    load_default_split_manifest,
    load_processed_mc0_corpus,
    resolve_smoke_question_ids_from_study,
    split_question_ids_from_manifest,
)


def resolve_corpus_context(
    study: StudyConfig,
    *,
    corpus_jsonl_paths: Sequence[str | Path] | None = None,
    split_manifest_path: str | Path | None = None,
    corpus_root: str | Path | None = None,
    ro_seed: int = 42,
) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Load normalized corpus, split IDs, and smoke question IDs."""
    if split_manifest_path is not None:
        split_ids = split_question_ids_from_manifest(split_manifest_path)
    else:
        split_ids = load_default_split_manifest(corpus_root=corpus_root)
    corpus = load_processed_mc0_corpus(
        jsonl_paths=corpus_jsonl_paths,
        corpus_root=corpus_root,
        ro_seed=ro_seed,
    )
    smoke_ids = resolve_smoke_question_ids_from_study(
        smoke=study.run.smoke,
        split_question_ids=split_ids,
    )
    return corpus, split_ids, smoke_ids
