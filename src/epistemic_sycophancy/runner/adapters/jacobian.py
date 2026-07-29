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
from epistemic_sycophancy.stack.scales import scales_for_layer_feature_keys


def build_jacobian_fn(
    study: StudyConfig,
    stack: Any,
) -> Callable[..., Mapping[tuple[int, int], float]]:
    """Build ``(*, order_regime, question_ids) -> signed J`` via projected formula.

    Unit/toy stacks may expose ``fs_projection_batch`` returning residual
    gradients, latents, and question_ids for one layer. Production stacks use
    that same projection math once residual grads + latents are available;
    ``fs_projection_batch`` is the injectable batch surface for tests (DEC-065).
    """

    def jacobian_fn(
        *,
        order_regime: str,
        question_ids: Sequence[str],
    ) -> Mapping[tuple[int, int], float]:
        del order_regime  # batch provider / corpus path selects prompts by order
        qids = tuple(str(q) for q in question_ids)
        if not qids:
            raise ValueError("jacobian_fn requires nonempty question_ids")
        if not hasattr(stack, "fs_projection_batch"):
            raise ValueError(
                "build_jacobian_fn requires stack.fs_projection_batch for "
                "residual_gradients/latents (toy inject or production batch helper)"
            )
        batch = stack.fs_projection_batch(
            question_ids=qids,
            feature_chunk_size=int(study.run.feature_chunk_size),
            prompt_batch_size=int(study.run.prompt_batch_size),
        )
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
        keys = [(layer, fid) for fid in range(n_features)]
        scales = scales_for_layer_feature_keys(
            keys=keys,
            saes=stack.saes,
            scale_source="decoder_norm",
        )
        feature_scales = torch.tensor(
            list(scales),
            dtype=torch.float64,
            device=residual_gradients.device,
        )
        raw = project_residual_gradient(
            gradient=residual_gradients.to(dtype=torch.float64),
            decoder=decoder.to(dtype=torch.float64),
            feature_chunk_size=int(study.run.feature_chunk_size),
        )
        per_prompt = coefficient_jacobian(
            raw_projection=raw,
            latents=latents.to(dtype=torch.float64),
            feature_scales=feature_scales,
        )
        by_question: dict[str, list[torch.Tensor]] = {}
        for row, qid in enumerate(batch_qids):
            by_question.setdefault(qid, []).append(per_prompt[row].detach())
        # Restrict to requested IDs (equal question weight).
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
