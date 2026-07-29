"""Pinned real-model smoke helpers (Phase J REAL / DEC-043)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from epistemic_sycophancy.intervention.sae_delta import apply_additive_sae_delta
from epistemic_sycophancy.scoring.margins import margin_preference, truthful_margin


@dataclass(frozen=True)
class RealModelScoreBatch:
    """Scored A/B logits, margins, and labels for a tiny prompt batch."""

    logits: torch.Tensor  # [B, 2]
    margins: tuple[float, ...]
    labels: tuple[str, ...]


def _load_model_and_residuals(
    *,
    model_id: str,
    model_revision: str,
    prompts: Sequence[str],
    dtype: torch.dtype,
    seed: int,
):
    import transformers

    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    d_model = int(model.config.n_embd)
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    attention = encoded["attention_mask"]
    lengths = attention.sum(dim=1)
    residuals = torch.stack(
        [hidden[i, int(lengths[i].item()) - 1] for i in range(hidden.shape[0])],
        dim=0,
    ).to(dtype=dtype)
    return residuals, d_model


def score_real_model_batch(
    *,
    model_id: str,
    model_revision: str,
    prompts: Sequence[str],
    beta: Sequence[float],
    selected_indices: Sequence[int],
    scales: Sequence[float],
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    truthful_label: str = "A",
) -> RealModelScoreBatch:
    """Score A/B via a seeded linear head on last-token residuals (+ optional SAE)."""
    residuals, d_model = _load_model_and_residuals(
        model_id=model_id,
        model_revision=model_revision,
        prompts=prompts,
        dtype=dtype,
        seed=seed,
    )
    torch.manual_seed(seed)
    n_features = max(selected_indices) + 1 if selected_indices else 8
    n_features = max(n_features, 8)
    decoder = torch.randn(n_features, d_model, dtype=dtype)
    encoder = torch.randn(n_features, d_model, dtype=dtype)
    encoder_bias = torch.zeros(n_features, dtype=dtype)
    head = torch.randn(2, d_model, dtype=dtype)
    for param in (decoder, encoder, encoder_bias, head):
        param.requires_grad_(False)

    logits_rows = []
    margins = []
    labels = []
    for residual in residuals:
        intervened = apply_additive_sae_delta(
            residual=residual,
            selected_indices=list(selected_indices),
            scales=list(scales),
            beta=list(beta),
            encoder_weight=encoder,
            encoder_bias=encoder_bias,
            decoder_weight=decoder,
        )
        logits = head @ intervened
        score_a = float(logits[0].item())
        score_b = float(logits[1].item())
        margin = truthful_margin(
            score_a=score_a, score_b=score_b, truthful_label=truthful_label
        )
        logits_rows.append(logits)
        margins.append(margin)
        labels.append(margin_preference(margin))
    return RealModelScoreBatch(
        logits=torch.stack(logits_rows, dim=0),
        margins=tuple(margins),
        labels=tuple(labels),
    )
