"""Production score_fn adapter (ORCH-021): render MC0 + score_batch_through_hooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.prompts.render import render_mc0_subset
from epistemic_sycophancy.stack.scoring import score_batch_through_hooks


def build_score_fn(
    study: StudyConfig,
    stack: Any,
    *,
    corpus: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str = "CF",
    belief_condition: str = "N",
    install_hooks_cm: Any | None = None,
) -> Callable[[Sequence[str]], Mapping[str, float]]:
    """Build ``(question_ids) -> margins`` using stack LM logits at β=0 by default.

    When ``install_hooks_cm`` is None, scoring is unhooked (identity / baseline).
    Continuations come from ``study.experiment.continuation_A/B`` via tokenizer.encode.
    Microbatches rendered prompts by ``run.prompt_batch_size`` (DEC-090 amend / ADAPT-011).
    """
    tok = stack.tokenizer
    token_a = list(tok.encode(study.experiment.continuation_A, add_special_tokens=False))
    token_b = list(tok.encode(study.experiment.continuation_B, add_special_tokens=False))
    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError(
            "ORCH-021 requires single-token continuations; "
            f"got A={token_a!r}, B={token_b!r}"
        )
    prompt_batch_size = int(study.run.prompt_batch_size)
    if prompt_batch_size < 1:
        raise ValueError(
            f"run.prompt_batch_size must be positive; got {prompt_batch_size!r}"
        )

    # Coverage omitted → render requested question_ids only.
    def score_fn(question_ids: Sequence[str]) -> Mapping[str, float]:
        qids = tuple(str(q) for q in question_ids)
        qid_set = set(qids)
        rendered = render_mc0_subset(
            corpus_rows=corpus,
            question_ids=qids,
            split_question_ids=split_question_ids,
            order_regime=order_regime,
            belief_condition=belief_condition,
        )
        selected = [row for row in rendered if row.question_id in qid_set]
        missing = qid_set - {row.question_id for row in selected}
        if missing:
            raise ValueError(
                f"score_fn missing rendered prompts for question_ids={sorted(missing)}"
            )
        if not selected:
            return {}
        # Deduplicate same-hash neutrals before scoring (DEC-006 / DEC-079).
        uniq: list[Any] = []
        seen: set[str] = set()
        for row in selected:
            if row.question_id in seen:
                continue
            seen.add(row.question_id)
            uniq.append(row)

        def _score_microbatches() -> list[float]:
            margins: list[float] = []
            n = len(uniq)
            for start in range(0, n, prompt_batch_size):
                end = min(start + prompt_batch_size, n)
                slice_rows = uniq[start:end]
                batch = score_batch_through_hooks(
                    model=stack.model,
                    tokenizer=stack.tokenizer,
                    prompts=[row.text for row in slice_rows],
                    continuation_token_ids_A=token_a,
                    continuation_token_ids_B=token_b,
                    truthful_labels=tuple(row.truthful_label for row in slice_rows),
                    device=stack.device,
                    install_hooks_cm=None,
                )
                margins.extend(float(m) for m in batch.margins)
                # del batch
                # if stack.device.type == "cuda":
                #     torch.cuda.empty_cache()
            return margins

        if install_hooks_cm is None:
            all_margins = _score_microbatches()
        else:
            with install_hooks_cm:
                all_margins = _score_microbatches()

        return {
            row.question_id: margin
            for row, margin in zip(uniq, all_margins, strict=True)
        }

    return score_fn
