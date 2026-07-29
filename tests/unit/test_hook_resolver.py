"""RUN-005: Gemma-3 resid_post layer → hook module resolver."""

from __future__ import annotations

import torch.nn as nn
import pytest

from epistemic_sycophancy.stack.resolver import (
    UnknownHookLayerError,
    resolve_resid_post_module,
)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.linear = nn.Linear(4, 4, bias=False)


class _FakeTextModel(nn.Module):
    def __init__(self, n_layers: int = 34) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeDecoderLayer(i) for i in range(n_layers)]
        )


class _FakeGemma3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeTextModel()


class _FakeGemma3ForConditionalGeneration(nn.Module):
    """Minimal tree matching Gemma3ForConditionalGeneration naming."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeGemma3Model()


@pytest.mark.unit
def test_stack__resid_post_resolver__maps_layer_to_hook_module() -> None:
    """RUN-005: layer L → model.language_model.layers[L]; unknown raises."""
    root = _FakeGemma3ForConditionalGeneration()
    for layer in (9, 17, 22, 29):
        module = resolve_resid_post_module(
            root,
            layer=layer,
            resolver_id="gemma3_resid_post",
        )
        assert module is root.model.language_model.layers[layer]
        assert module.layer_idx == layer

    with pytest.raises(UnknownHookLayerError):
        resolve_resid_post_module(
            root,
            layer=99,
            resolver_id="gemma3_resid_post",
        )

    with pytest.raises(ValueError):
        resolve_resid_post_module(
            root,
            layer=9,
            resolver_id="unknown_resolver",
        )
