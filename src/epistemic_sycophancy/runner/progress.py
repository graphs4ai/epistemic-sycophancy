"""Adam-step prompt-microbatch progress bars (ORCH-LOG-007b/c / DEC-092)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

from tqdm import tqdm

from epistemic_sycophancy.config.study import StudySmokeConfig
from epistemic_sycophancy.prompts.render import render_mc0_subset

_ACTIVE_ADAM_STEP: ContextVar["_AdamStepBatchProgress | None"] = ContextVar(
    "adam_step_batch_progress",
    default=None,
)


class _AdamStepBatchProgress:
    """Handle around a per-step tqdm bar with a fixed pre-computed total."""

    def __init__(self, bar: Any) -> None:
        self.bar = bar

    def tick(self, n: int = 1) -> None:
        self.bar.update(int(n))


def n_prompt_microbatches(n_rows: int, *, batch_size: int) -> int:
    """Return ``ceil(n_rows / batch_size)`` (0 when no rows)."""
    if n_rows < 1:
        return 0
    size = int(batch_size)
    if size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")
    return (int(n_rows) + size - 1) // size


def count_belief_condition_rows(
    *,
    corpus: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    question_ids: Sequence[str],
    order_regime: str,
    belief_condition: str,
) -> int:
    """Count rendered rows for one belief condition (N deduped 1/q; IB/CB keep variants)."""
    qids = tuple(str(q) for q in question_ids)
    if not qids:
        return 0
    smoke = StudySmokeConfig(question_ids=qids)
    rendered = render_mc0_subset(
        corpus_rows=corpus,
        smoke=smoke,
        split_question_ids=split_question_ids,
        order_regime=order_regime,
        belief_condition=belief_condition,
    )
    if belief_condition == "N":
        seen: set[str] = set()
        n = 0
        for row in rendered:
            if row.question_id in seen:
                continue
            seen.add(row.question_id)
            n += 1
        return n
    return len(list(rendered))


def count_adam_step_prompt_microbatches(
    *,
    corpus: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    question_ids: Sequence[str],
    order_regime: str,
    prompt_batch_size: int,
) -> int:
    """Fixed prompt-microbatch total for one Adam step (grad + logged objective).

    Call graph per step (DEC-076 / DEC-084):
    - ``grad_fn``: ``build_margin_payload`` (2·N + IB + 2·CB scorers) + jac (N+IB+CB)
    - ``objective_fn``: ``build_margin_payload`` again (2·N + IB + 2·CB)

    Therefore total microbatches = ``5·bN + 3·bIB + 5·bCB`` where
    ``bX = ceil(n_rows(X) / prompt_batch_size)``.
    """
    batch_size = int(prompt_batch_size)
    counts = {
        belief: count_belief_condition_rows(
            corpus=corpus,
            split_question_ids=split_question_ids,
            question_ids=question_ids,
            order_regime=order_regime,
            belief_condition=belief,
        )
        for belief in ("N", "IB", "CB")
    }
    b_n = n_prompt_microbatches(counts["N"], batch_size=batch_size)
    b_ib = n_prompt_microbatches(counts["IB"], batch_size=batch_size)
    b_cb = n_prompt_microbatches(counts["CB"], batch_size=batch_size)
    return 5 * b_n + 3 * b_ib + 5 * b_cb


@contextmanager
def adam_step_batch_progress(
    *,
    step: int,
    n_steps: int,
    total: int,
) -> Iterator[Any]:
    """Open a tqdm bar for one Adam step with a fixed pre-computed ``total``."""
    fixed_total = max(0, int(total))
    bar = tqdm(
        total=fixed_total,
        desc=f"adam step {int(step) + 1}/{int(n_steps)}",
        unit="batch",
        leave=True,
    )
    handle = _AdamStepBatchProgress(bar)
    token: Token = _ACTIVE_ADAM_STEP.set(handle)
    try:
        yield bar
    finally:
        _ACTIVE_ADAM_STEP.reset(token)
        bar.close()


def tick_prompt_batch(n: int = 1) -> None:
    """Advance the active Adam step bar after one prompt microbatch completes."""
    handle = _ACTIVE_ADAM_STEP.get()
    if handle is not None:
        handle.tick(n)
