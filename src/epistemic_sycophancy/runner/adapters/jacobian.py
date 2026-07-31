"""Production jacobian_fn adapter (ORCH-022 / DEC-060)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from epistemic_sycophancy.config.study import StudyConfig
from epistemic_sycophancy.feature_selection.projected_gradient import (
    coefficient_jacobian,
    project_residual_gradient,
    question_macro_jacobian,
)
from epistemic_sycophancy.feature_selection.components import (
    selection_component_prompts,
)
from epistemic_sycophancy.metrics.baseline_partition import BaselinePartition
from epistemic_sycophancy.prompts.render import RenderedPromptRow, render_mc0_subset
from epistemic_sycophancy.runner.adapters.fs_batch import compute_fs_projection_batch


def build_jacobian_fn(
    study: StudyConfig,
    stack: Any,
    *,
    corpus: Sequence[Mapping[str, object]] | None = None,
    split_question_ids: Mapping[str, Sequence[str]] | None = None,
) -> Callable[..., Mapping[tuple[int, int], float]]:
    """Build ``(*, order_regime, question_ids, component=) -> signed J``.

    Unit/toy stacks may expose ``fs_projection_batch``. Production stacks use
    multi-condition render + ``selection_component_prompts`` (DEC-085).
    """

    def jacobian_fn(
        *,
        order_regime: str,
        question_ids: Sequence[str],
        component: str = "resistance",
    ) -> Mapping[tuple[int, int], float]:
        qids = tuple(str(q) for q in question_ids)
        if not qids:
            raise ValueError("jacobian_fn requires nonempty question_ids")

        if hasattr(stack, "fs_projection_batch"):
            batch_kwargs: dict[str, Any] = {
                "question_ids": qids,
                "feature_chunk_size": int(study.run.feature_chunk_size),
                "prompt_batch_size": int(study.run.prompt_batch_size),
                "order_regime": order_regime,
            }
            # Toy injectors may ignore component; production fakes may use it.
            try:
                batch = stack.fs_projection_batch(**batch_kwargs, component=component)
            except TypeError:
                batch = stack.fs_projection_batch(**batch_kwargs)
        else:
            if corpus is None or split_question_ids is None:
                raise ValueError(
                    "build_jacobian_fn requires stack.fs_projection_batch or "
                    "corpus+split_question_ids for production projection (ORCH-036)"
                )
            batch = _production_component_batch(
                study=study,
                stack=stack,
                corpus=corpus,
                split_question_ids=split_question_ids,
                order_regime=order_regime,
                question_ids=qids,
                component=component,
            )
            if batch is None:
                return {}

        layer = int(batch["layer"])
        residual_gradients = batch["residual_gradients"]
        latents = batch["latents"]
        batch_qids = [str(q) for q in batch["question_ids"]]
        if len(batch_qids) != residual_gradients.shape[0]:
            raise ValueError("fs_projection_batch question_ids length must match batch")
        sae = stack.saes[layer]
        decoder = (
            sae.decoder_weight
            if hasattr(sae, "decoder_weight")
            else sae.W_dec
        )
        n_features = int(decoder.shape[0])
        decoder_f64 = decoder.detach().to(dtype=torch.float64)
        # Full-width decoder norms in one vectorized op (layer17 limited path).
        feature_scales = torch.linalg.vector_norm(decoder_f64, dim=1)
        if not bool(torch.all(feature_scales > 0)):
            bad = int((feature_scales <= 0).nonzero()[0].item())
            raise ValueError(f"decoder_norm must be > 0; feature_id={bad} got 0")
        raw = project_residual_gradient(
            gradient=residual_gradients.to(dtype=torch.float64),
            decoder=decoder_f64,
            feature_chunk_size=int(study.run.feature_chunk_size),
        )
        per_prompt = coefficient_jacobian(
            raw_projection=raw,
            latents=latents.to(dtype=torch.float64),
            feature_scales=feature_scales,
        )
        if batch.get("weights_applied"):
            # §11.3 / FSC-003: weights already in residual grads — sum once.
            macro = per_prompt.sum(dim=0)
        else:
            by_question: dict[str, list[torch.Tensor]] = {}
            for row, qid in enumerate(batch_qids):
                by_question.setdefault(qid, []).append(per_prompt[row].detach())
            filtered = {qid: by_question[qid] for qid in qids if qid in by_question}
            missing = set(qids) - set(filtered)
            if missing:
                raise ValueError(
                    f"jacobian_fn missing projection rows for question_ids={sorted(missing)}"
                )
            macro = question_macro_jacobian(filtered)
        return {
            (layer, fid): float(macro[fid].item()) for fid in range(n_features)
        }

    return jacobian_fn


def render_fs_multi_condition_rows(
    *,
    corpus_rows: Sequence[Mapping[str, object]],
    question_ids: Sequence[str],
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str,
) -> dict[str, tuple[RenderedPromptRow, ...]]:
    """Render N/IB/CB on the FS coverage question set (DEC-085 / FSC-001).

    Neutrals are deduplicated to one row per question_id. IB and CB keep every
    variant. Optimization/holdout rows are never selected via these IDs.
    """
    by_condition: dict[str, tuple[RenderedPromptRow, ...]] = {}
    for belief in ("N", "IB", "CB"):
        rendered = render_mc0_subset(
            corpus_rows=corpus_rows,
            question_ids=question_ids,
            split_question_ids=split_question_ids,
            order_regime=order_regime,
            belief_condition=belief,
        )
        if belief == "N":
            uniq: list[RenderedPromptRow] = []
            seen: set[str] = set()
            for row in rendered:
                if row.question_id in seen:
                    continue
                seen.add(row.question_id)
                uniq.append(row)
            by_condition[belief] = tuple(uniq)
        else:
            by_condition[belief] = tuple(rendered)
    return by_condition


def select_fs_component_prompt_rows(
    *,
    component: str,
    by_condition: Mapping[str, Sequence[RenderedPromptRow]],
    partition: BaselinePartition,
) -> tuple[RenderedPromptRow, ...]:
    """Filter multi-condition FS rows for one §11.2 component (FSC-002 / FEAT-023)."""
    flat: list[RenderedPromptRow] = []
    for rows in by_condition.values():
        flat.extend(rows)
    return selection_component_prompts(
        component=component,
        prompt_rows=flat,
        partition=partition,
    )


def _load_fs_partition(
    *,
    study: StudyConfig,
    order_regime: str,
) -> BaselinePartition:
    """Load frozen FS-split baseline partition for ``order_regime`` (DEC-085)."""
    import json
    from pathlib import Path

    path = Path(study.run.artifact_dir) / "baseline" / f"partition_{order_regime}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"FS component Jacobians require frozen partition at {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BaselinePartition(
        order_regime=str(payload.get("order_regime", order_regime)),
        q_plus=frozenset(str(q) for q in payload["q_plus"]),
        q_minus=frozenset(str(q) for q in payload["q_minus"]),
        q_tie=frozenset(str(q) for q in payload.get("q_tie", ())),
        n_q_tie=int(payload.get("n_q_tie", len(payload.get("q_tie", ())))),
    )


def _production_component_batch(
    *,
    study: StudyConfig,
    stack: Any,
    corpus: Sequence[Mapping[str, object]],
    split_question_ids: Mapping[str, Sequence[str]],
    order_regime: str,
    question_ids: Sequence[str],
    component: str,
) -> dict[str, Any] | None:
    """Render N/IB/CB, select component rows, project φ grads (DEC-085)."""
    by_condition = render_fs_multi_condition_rows(
        corpus_rows=corpus,
        question_ids=tuple(question_ids),
        split_question_ids=split_question_ids,
        order_regime=order_regime,
    )
    partition = _load_fs_partition(study=study, order_regime=order_regime)
    selected = select_fs_component_prompt_rows(
        component=component,
        by_condition=by_condition,
        partition=partition,
    )
    if not selected:
        return None
    tok = stack.tokenizer
    token_a = list(tok.encode(study.experiment.continuation_A, add_special_tokens=False))
    token_b = list(tok.encode(study.experiment.continuation_B, add_special_tokens=False))
    layer = int(study.stack.sae.layers[0])
    return compute_fs_projection_batch(
        stack,
        layer=layer,
        texts=[r.text for r in selected],
        question_ids=[r.question_id for r in selected],
        continuation_token_ids_A=token_a,
        continuation_token_ids_B=token_b,
        truthful_labels=[r.truthful_label for r in selected],
        tau=float(study.experiment.tau),
        prompt_batch_size=int(study.run.prompt_batch_size),
    )
