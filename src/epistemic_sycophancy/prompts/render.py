"""WIRE-005: render MC0 prompts for StudyConfig FS coverage subset."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.config.schema import InvalidExperimentConfig
from epistemic_sycophancy.config.study import StudyFsCoverageConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.prompts.templates import (
    StructuredPrompt,
    assert_belief_text_has_no_label_artifacts,
)

_FS_SPLIT = "feature_selection"
_ALLOWED_SPLITS = frozenset({"feature_selection", "optimization"})


@dataclass(frozen=True)
class RenderedPromptRow:
    """One rendered MC0 prompt tied to a corpus question."""

    question_id: str
    split: str
    order_regime: str
    belief_condition: str
    truthful_label: str
    text: str

    @property
    def condition(self) -> str:
        """Alias for ``belief_condition`` (FEAT-010 / ``selection_component_prompts``)."""
        return self.belief_condition


def select_coverage_question_ids(
    *,
    coverage: StudyFsCoverageConfig | None,
    split_question_ids: Mapping[str, Sequence[str]],
    split: str = _FS_SPLIT,
) -> tuple[str, ...]:
    """Resolve coverage allowlist, seeded take-N, or full split (default).

    ``None`` / empty coverage → all IDs in ``split``. Never holdout.
    """
    if split not in _ALLOWED_SPLITS:
        raise HoldoutAccessError(
            f"coverage split must be one of {sorted(_ALLOWED_SPLITS)}; got {split!r}"
        )
    if split not in split_question_ids:
        raise InvalidExperimentConfig(
            f"coverage split {split!r} missing from split_question_ids"
        )
    sorted_ids = sorted(str(qid) for qid in split_question_ids[split])

    if coverage is None or (
        coverage.question_ids is None
        and coverage.n_questions is None
        and coverage.seed is None
    ):
        return tuple(sorted_ids)

    if coverage.question_ids is not None:
        return tuple(coverage.question_ids)

    assert coverage.n_questions is not None and coverage.seed is not None
    if coverage.n_questions > len(sorted_ids):
        raise InvalidExperimentConfig(
            f"fs_coverage n_questions={coverage.n_questions} exceeds "
            f"|{split}|={len(sorted_ids)}"
        )
    if not sorted_ids:
        return ()
    digest = hashlib.sha256(f"{coverage.seed}".encode()).digest()
    offset = int.from_bytes(digest[:4], "big") % len(sorted_ids)
    rotated = sorted_ids[offset:] + sorted_ids[:offset]
    return tuple(rotated[: coverage.n_questions])


def render_mc0_text(prompt: StructuredPrompt) -> str:
    """Render a StructuredPrompt into a bare MC0 string (DEC-010 continuations)."""
    if prompt.format != "MC0":
        raise ValueError(f"expected format MC0; got {prompt.format!r}")
    if prompt.belief_context is not None:
        assert_belief_text_has_no_label_artifacts(
            prompt.belief_context, answer_suffix=prompt.instruction or None
        )
        belief_block = f"{prompt.belief_context}\n"
    else:
        belief_block = ""
    body = (
        f"{belief_block}"
        f"Question: {prompt.question_text}\n"
        f"A. {prompt.candidate_a}\n"
        f"B. {prompt.candidate_b}\n"
        f"{prompt.instruction}"
        f"{prompt.suffix}"
    )
    return body


def render_mc0_subset(
    *,
    corpus_rows: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str,
    coverage: StudyFsCoverageConfig | None = None,
    question_ids: Sequence[str] | None = None,
    belief_condition: str = "N",
) -> tuple[RenderedPromptRow, ...]:
    """Select coverage questions and render MC0 neutral (or conditioned) prompts.

    Prefer ``question_ids`` when provided; otherwise resolve from ``coverage``
    (omit / None → full feature_selection split).
    """
    if question_ids is not None:
        selected = set(str(q) for q in question_ids)
    else:
        selected = set(
            select_coverage_question_ids(
                coverage=coverage, split_question_ids=split_question_ids
            )
        )
    rendered: list[RenderedPromptRow] = []
    for row in corpus_rows:
        qid = str(row["question_id"])
        if qid not in selected:
            continue
        split = str(row["split"])
        if split not in _ALLOWED_SPLITS:
            raise HoldoutAccessError(f"corpus row split {split!r} is forbidden")
        if str(row.get("order_regime", order_regime)) != order_regime:
            continue
        if str(row.get("belief_condition", "N")) != belief_condition:
            continue
        structured = StructuredPrompt(
            question_text=str(row["question_text"]),
            candidate_a=str(row["candidate_a"]),
            candidate_b=str(row["candidate_b"]),
            instruction=str(row.get("instruction", "Answer with A or B.")),
            suffix=str(row.get("suffix", "")),
            prompt_template_version=str(row.get("prompt_template_version", "v1")),
            format=str(row.get("format", "MC0")),
            order_regime=order_regime,
            belief_condition=belief_condition,
            belief_context=(
                None
                if row.get("belief_context") is None
                else str(row["belief_context"])
            ),
        )
        rendered.append(
            RenderedPromptRow(
                question_id=qid,
                split=split,
                order_regime=order_regime,
                belief_condition=belief_condition,
                truthful_label=str(row.get("truthful_label", "A")),
                text=render_mc0_text(structured),
            )
        )
    rendered.sort(key=lambda r: r.question_id)
    missing = selected - {r.question_id for r in rendered}
    if missing:
        raise InvalidExperimentConfig(
            f"coverage questions missing from corpus for "
            f"order={order_regime!r} belief={belief_condition!r}: {sorted(missing)}"
        )
    return tuple(rendered)
