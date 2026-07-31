"""GRAD-012: optimize margin projection honors run.prompt_batch_size (DEC-090)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from epistemic_sycophancy.runner.adapters.margin_batch import (
    compute_margin_projection_batch,
)


class _LayerMod(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ToyCausalLM(nn.Module):
    """Minimal Gemma3-shaped model: resid_post hook + logits from residual."""

    def __init__(self, *, d_model: int = 4, n_layers: int = 18, vocab: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        layers = nn.ModuleList([_LayerMod() for _ in range(n_layers)])
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=layers)
        )
        # Map last-token residual → vocab logits (float64 for DEC-022).
        self.unembed = nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                ],
                dtype=torch.float64,
            )[:vocab, :d_model]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del kwargs
        batch, seq = input_ids.shape
        # Residuals keyed by token id so microbatches match full-batch order.
        # Fill all positions uniformly (avoid writing only at padded index -1).
        residual = torch.zeros(batch, seq, self.d_model, dtype=torch.float64)
        for i in range(batch):
            seed = float(input_ids[i, 0].item())
            residual[i, :, 0] = 0.5 + 0.1 * seed
            residual[i, :, 1] = 0.25 - 0.05 * seed
            residual[i, :, 2] = 1.0 + 0.2 * seed
            residual[i, :, 3] = 0.1 * seed
        residual = self.model.language_model.layers[17](residual)
        logits = residual @ self.unembed.T
        del attention_mask
        return SimpleNamespace(logits=logits)


class _Tok:
    def __call__(self, texts, return_tensors="pt", padding=True):
        batch = len(texts)
        # Content-stable ids + uneven lengths (padding) across microbatches.
        lengths = [2 + (sum(ord(c) for c in t) % 2) for t in texts]
        max_len = max(lengths) if lengths else 1
        input_ids = torch.zeros(batch, max_len, dtype=torch.long)
        attention = torch.zeros(batch, max_len, dtype=torch.long)
        for i, (text, length) in enumerate(zip(texts, lengths, strict=True)):
            code = (sum(ord(c) for c in text) % 97) + 1
            attention[i, :length] = 1
            input_ids[i, :length] = code
        return {"input_ids": input_ids, "attention_mask": attention}


class _ToySae:
    def __init__(self, *, d_model: int = 4, n_features: int = 3) -> None:
        torch.manual_seed(0)
        # sae-lens layout: W_enc [d_model, n_features] for work @ W_enc.
        self.W_enc = torch.randn(d_model, n_features, dtype=torch.float64)
        self.b_enc = torch.zeros(n_features, dtype=torch.float64)
        self.threshold = torch.full((n_features,), 0.01, dtype=torch.float64)
        self.W_dec = torch.randn(n_features, d_model, dtype=torch.float64)


class _Stack:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = _Tok()
        self.model = _ToyCausalLM()
        self.config = SimpleNamespace(
            hooks=SimpleNamespace(resolver_id="gemma3_resid_post")
        )
        self.saes = {17: SimpleNamespace(sae=_ToySae())}


@pytest.mark.unit
def test_margin_batch__prompt_microbatches__match_full_batch_grads_and_latents() -> None:
    """GRAD-012: prompt_batch_size=1 and uneven 2 match full-batch ∂M/∂x + latents."""
    stack = _Stack()
    texts = ("p0", "p1", "p2")
    question_ids = ("q0", "q1", "q2")
    labels = ("A", "B", "A")
    kwargs = dict(
        stack=stack,
        layer=17,
        texts=texts,
        question_ids=question_ids,
        continuation_token_ids_A=(0,),
        continuation_token_ids_B=(1,),
        truthful_labels=labels,
    )

    full = compute_margin_projection_batch(**kwargs, prompt_batch_size=3)
    for chunk in (1, 2):
        micro = compute_margin_projection_batch(**kwargs, prompt_batch_size=chunk)
        assert micro["question_ids"] == list(question_ids)
        assert micro["layer"] == 17
        # DEC-022 float64 grads; latents use production float32 encode
        # (work @ W_enc), so tolerate float32 GEMM reduction vs full batch.
        assert torch.allclose(
            micro["residual_gradients"],
            full["residual_gradients"],
            atol=1e-10,
            rtol=1e-9,
        )
        assert torch.allclose(
            micro["latents"],
            full["latents"],
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            micro["decoder"],
            full["decoder"],
            atol=1e-10,
            rtol=1e-9,
        )
        assert torch.allclose(
            micro["feature_scales"],
            full["feature_scales"],
            atol=1e-10,
            rtol=1e-9,
        )
