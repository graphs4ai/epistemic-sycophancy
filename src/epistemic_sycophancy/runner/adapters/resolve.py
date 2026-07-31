"""Shared corpus resolution for dispatch defaults (ORCH-027+)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.runner.adapters.corpus import (
    _HOLDUT_SPLITS,
    load_default_split_manifest,
    load_processed_mc0_corpus,
    resolve_fs_coverage_question_ids,
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
    """Load normalized corpus, split IDs, and FS coverage question IDs.

    Split IDs are the intersection of the split manifest and question IDs
    actually present in the loaded MC0 corpus (avoids manifest orphans).
    Coverage omitted → full feature_selection split present in corpus.
    """
    if split_manifest_path is not None:
        manifest_ids = split_question_ids_from_manifest(split_manifest_path)
    else:
        manifest_ids = load_default_split_manifest(corpus_root=corpus_root)
    corpus = load_processed_mc0_corpus(
        jsonl_paths=corpus_jsonl_paths,
        corpus_root=corpus_root,
        ro_seed=ro_seed,
    )
    present: dict[str, set[str]] = defaultdict(set)
    for row in corpus:
        present[str(row["split"])].add(str(row["question_id"]))
    split_ids: dict[str, tuple[str, ...]] = {}
    for split, ids in manifest_ids.items():
        if split in _HOLDUT_SPLITS or str(split).startswith("holdout"):
            split_ids[split] = tuple(ids)
            continue
        kept = tuple(qid for qid in ids if qid in present.get(split, set()))
        # If manifest/corpus disagree, fall back to corpus-present IDs for the split.
        split_ids[split] = kept if kept else tuple(sorted(present.get(split, ())))
    coverage_ids = resolve_fs_coverage_question_ids(
        coverage=study.run.fs_coverage,
        split_question_ids=split_ids,
    )
    return corpus, split_ids, coverage_ids
