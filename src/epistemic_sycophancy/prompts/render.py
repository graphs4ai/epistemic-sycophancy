"""WIRE-005: render MC0 prompts for StudyConfig smoke subset."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.config.schema import InvalidExperimentConfig
from epistemic_sycophancy.config.study import StudySmokeConfig
from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.prompts.templates import (
    StructuredPrompt,
    assert_belief_text_has_no_label_artifacts,
)

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


def select_smoke_question_ids(
    *,
    smoke: StudySmokeConfig,
    split_question_ids: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Resolve smoke allowlist or seeded first-N (DEC-059). Never holdout."""
    if smoke.question_ids is not None:
        return tuple(smoke.question_ids)
    assert smoke.split is not None and smoke.n_questions is not None
    assert smoke.seed is not None
    if smoke.split not in _ALLOWED_SPLITS:
        raise HoldoutAccessError(
            f"smoke split must be one of {sorted(_ALLOWED_SPLITS)}; got {smoke.split!r}"
        )
    if smoke.split not in split_question_ids:
        raise InvalidExperimentConfig(
            f"smoke split {smoke.split!r} missing from split_question_ids"
        )
    sorted_ids = sorted(str(qid) for qid in split_question_ids[smoke.split])
    if smoke.n_questions > len(sorted_ids):
        raise InvalidExperimentConfig(
            f"smoke n_questions={smoke.n_questions} exceeds "
            f"|{smoke.split}|={len(sorted_ids)}"
        )
    # Deterministic: sort, then rotate by seed hash offset, take first N.
    if not sorted_ids:
        return ()
    digest = hashlib.sha256(f"{smoke.seed}".encode()).digest()
    offset = int.from_bytes(digest[:4], "big") % len(sorted_ids)
    rotated = sorted_ids[offset:] + sorted_ids[:offset]
    return tuple(rotated[: smoke.n_questions])


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
    smoke: StudySmokeConfig,
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str,
    belief_condition: str = "N",
) -> tuple[RenderedPromptRow, ...]:
    """Select smoke questions and render MC0 neutral (or conditioned) prompts."""
    selected = set(
        select_smoke_question_ids(smoke=smoke, split_question_ids=split_question_ids)
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
            f"smoke questions missing from corpus for "
            f"order={order_regime!r} belief={belief_condition!r}: {sorted(missing)}"
        )
    return tuple(rendered)
