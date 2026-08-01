"""Production margin_payload adapter (ORCH-023 / DEC-076 / PERF-BASELINE)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from epistemic_sycophancy.config.study import StudyConfig


@dataclass(frozen=True)
class FrozenMarginBaselines:
    neutral: Mapping[str, float]
    contextual: Mapping[str, tuple[float, ...]]


@dataclass
class MarginBaselineCache:
    scorer: Callable[..., Mapping[str, Any]]
    coefficient_length: int
    _entries: dict[
        tuple[str, ...],
        FrozenMarginBaselines,
    ] = field(default_factory=dict)

    def get(
        self,
        question_ids: Sequence[str],
    ) -> FrozenMarginBaselines:
        qids = tuple(str(q) for q in question_ids)

        cached = self._entries.get(qids)
        if cached is not None:
            return cached

        zero_beta = (0.0,) * self.coefficient_length

        neutral = {
            str(qid): float(value)
            for qid, value in self.scorer(
                belief_condition="N",
                question_ids=qids,
                beta=zero_beta,
            ).items()
        }

        contextual = {
            str(qid): tuple(float(x) for x in _as_sequence(values))
            for qid, values in self.scorer(
                belief_condition="CB",
                question_ids=qids,
                beta=zero_beta,
            ).items()
        }

        result = FrozenMarginBaselines(
            neutral=neutral,
            contextual=contextual,
        )
        self._entries[qids] = result
        return result


def build_margin_payload(
    study: StudyConfig,
    stack: Any,
    *,
    beta: Sequence[float],
    question_ids: Sequence[str],
    partitions: Mapping[str, Any],
    margin_scorer: Callable[..., Mapping[str, Any]] | None = None,
    order_regime: str = "CF",
    baseline_cache: MarginBaselineCache | None = None,
) -> dict[str, Any]:
    """Build optimize margin payload via live scoring at ``beta``.

    ``margin_scorer(belief_condition=, question_ids=, beta=)`` returns either
    scalars (N) or variant sequences (IB/CB) keyed by question_id. When None,
    requires ``stack.score_belief_margins`` (production / richer fakes).

    When ``baseline_cache`` is provided, frozen N/CB margins at β=0 are memoized
    per question-ID tuple (PERF-BASELINE). Current-β margins remain live.
    """
    del study
    # order_regime is applied by the closed-over margin_scorer (DEC-087).
    del order_regime
    qids = tuple(str(q) for q in question_ids)
    scorer = margin_scorer
    if scorer is None:
        if not hasattr(stack, "score_belief_margins"):
            raise ValueError(
                "build_margin_payload requires margin_scorer or "
                "stack.score_belief_margins (DEC-076 live scoring)"
            )
        scorer = stack.score_belief_margins

    beta_t = tuple(float(b) for b in beta)
    current_n = dict(scorer(belief_condition="N", question_ids=qids, beta=beta_t))
    if baseline_cache is None:
        # Existing compatibility path (ORCH-023 / DEC-076).
        baseline_n = dict(
            scorer(
                belief_condition="N",
                question_ids=qids,
                beta=tuple(0.0 for _ in beta_t) or (0.0,),
            )
        )
        if not beta_t:
            baseline_n = dict(
                scorer(belief_condition="N", question_ids=qids, beta=(0.0,))
            )
        baseline_cb = {
            qid: tuple(float(x) for x in _as_sequence(vals))
            for qid, vals in dict(
                scorer(
                    belief_condition="CB",
                    question_ids=qids,
                    beta=tuple(0.0 for _ in beta_t) if beta_t else (0.0,),
                )
            ).items()
        }
    else:
        baselines = baseline_cache.get(qids)
        baseline_n = baselines.neutral
        baseline_cb = baselines.contextual
    ib = {
        qid: tuple(float(x) for x in _as_sequence(vals))
        for qid, vals in dict(
            scorer(belief_condition="IB", question_ids=qids, beta=beta_t)
        ).items()
    }
    cb = {
        qid: tuple(float(x) for x in _as_sequence(vals))
        for qid, vals in dict(
            scorer(belief_condition="CB", question_ids=qids, beta=beta_t)
        ).items()
    }
    q_plus = frozenset(str(q) for q in partitions["q_plus"])
    q_minus = frozenset(str(q) for q in partitions["q_minus"])
    return {
        "ib_margins_by_question": ib,
        "cb_margins_by_question": cb,
        "baseline_cb_margins": baseline_cb,
        "baseline_neutral_margins": {
            qid: float(v) for qid, v in baseline_n.items()
        },
        "current_neutral_margins": {
            qid: float(v) for qid, v in current_n.items()
        },
        "q_plus": q_plus,
        "q_minus": q_minus,
    }


def _as_sequence(value: Any) -> Sequence[float]:
    if isinstance(value, (list, tuple)):
        return value
    return (float(value),)
