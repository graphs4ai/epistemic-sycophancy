"""Production belief-margin scorer for live objective adapters (DEC-076)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig, StudySmokeConfig
from epistemic_sycophancy.prompts.render import render_mc0_subset
from epistemic_sycophancy.stack.scoring import score_batch_through_hooks


def build_belief_margin_scorer(
    study: StudyConfig,
    stack: Any,
    *,
    corpus: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str = "CF",
):
    """Return ``score_belief_margins(belief_condition=, question_ids=, beta=)``.

    Live stack scoring (DEC-076): β=0 is unhooked; nonzero β installs hooks with
    per-batch ``prompt_lengths``.
    """
    tok = stack.tokenizer
    token_a = list(tok.encode(study.experiment.continuation_A, add_special_tokens=False))
    token_b = list(tok.encode(study.experiment.continuation_B, add_special_tokens=False))
    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError(
            "belief scorer requires single-token continuations; "
            f"got A={token_a!r}, B={token_b!r}"
        )
    feature_ids = tuple(
        (int(f[0]), int(f[1]))
        if isinstance(f, (list, tuple))
        else (int(study.stack.sae.layers[0]), int(f))
        for f in study.experiment.feature_ids
    )
    scales = tuple(float(s) for s in study.experiment.feature_scales)

    def score_belief_margins(
        *,
        belief_condition: str,
        question_ids: Sequence[str],
        beta: Sequence[float],
        order_regime: str = order_regime,
    ) -> Mapping[str, Any]:
        qids = tuple(str(q) for q in question_ids)
        if not qids:
            return {}
        smoke = StudySmokeConfig(question_ids=qids)
        rendered = render_mc0_subset(
            corpus_rows=corpus,
            smoke=smoke,
            split_question_ids=split_question_ids,
            order_regime=order_regime,
            belief_condition=belief_condition,
        )
        if belief_condition == "N":
            rows: list[Any] = []
            seen: set[str] = set()
            for row in rendered:
                if row.question_id in seen:
                    continue
                seen.add(row.question_id)
                rows.append(row)
        else:
            rows = list(rendered)

        beta_t = tuple(float(b) for b in beta)
        hooks_cm = None
        if feature_ids and any(abs(b) > 0.0 for b in beta_t):
            encoded = tok(
                [row.text for row in rows],
                return_tensors="pt",
                padding=True,
            )
            if "attention_mask" in encoded:
                lengths = tuple(
                    int(x) for x in encoded["attention_mask"].sum(dim=-1).tolist()
                )
            else:
                lengths = tuple(int(encoded["input_ids"].shape[1]) for _ in rows)
            hooks_cm = stack.install_hooks(
                selected_keys=feature_ids,
                scales=scales,
                beta=beta_t,
                prompt_lengths=lengths,
            )

        batch = score_batch_through_hooks(
            model=stack.model,
            tokenizer=stack.tokenizer,
            prompts=[row.text for row in rows],
            continuation_token_ids_A=token_a,
            continuation_token_ids_B=token_b,
            truthful_labels=tuple(row.truthful_label for row in rows),
            device=stack.device,
            install_hooks_cm=hooks_cm,
        )
        margins = tuple(float(m) for m in batch.margins)
        if belief_condition == "N":
            return {
                row.question_id: margin
                for row, margin in zip(rows, margins, strict=True)
            }
        by_q: dict[str, list[float]] = defaultdict(list)
        for row, margin in zip(rows, margins, strict=True):
            by_q[row.question_id].append(margin)
        return {qid: tuple(vals) for qid, vals in by_q.items()}

    return score_belief_margins
