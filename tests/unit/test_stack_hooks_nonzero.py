"""WIRE-001: nonzero-β JumpReLU delta wired into multi-layer hooks."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

from epistemic_sycophancy.sae.jumprelu_delta import apply_additive_jumprelu_sae_delta
from epistemic_sycophancy.stack.hooks import install_multi_layer_hooks


class _FakeModule:
    def __init__(self) -> None:
        self._hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> Any:
        self._hooks.append(hook)

        def remove() -> None:
            self._hooks.remove(hook)

        return SimpleNamespace(remove=remove)

    def run(self, activation: torch.Tensor) -> torch.Tensor:
        output: Any = activation
        for hook in list(self._hooks):
            output = hook(self, (), output)
        return output


@contextmanager
def _patch_resolver(modules: dict[int, _FakeModule]) -> Iterator[None]:
    import epistemic_sycophancy.stack.hooks as hooks_mod

    original = hooks_mod.resolve_resid_post_module

    def fake_resolve(model: Any, *, layer: int, resolver_id: str) -> _FakeModule:
        del model, resolver_id
        return modules[layer]

    hooks_mod.resolve_resid_post_module = fake_resolve  # type: ignore[assignment]
    try:
        yield
    finally:
        hooks_mod.resolve_resid_post_module = original  # type: ignore[assignment]


def _fake_sae(
    *,
    d_model: int,
    n_features: int,
    encoder_weight: torch.Tensor,
    encoder_bias: torch.Tensor,
    threshold: torch.Tensor,
    decoder_weight: torch.Tensor,
) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
        d_model=d_model,
        d_sae=n_features,
    )


@pytest.mark.unit
def test_hooks__nonzero_beta__applies_jumprelu_delta_with_token_scope() -> None:
    """WIRE-001: nonzero β applies JumpReLU delta at last_prompt_token only."""
    torch.manual_seed(1)
    d_model = 4
    n_features = 5
    layer = 17
    encoder_weight = torch.randn(n_features, d_model, dtype=torch.float64)
    encoder_bias = torch.randn(n_features, dtype=torch.float64)
    threshold = torch.full((n_features,), 0.05, dtype=torch.float64)
    decoder_weight = torch.randn(n_features, d_model, dtype=torch.float64)
    sae = _fake_sae(
        d_model=d_model,
        n_features=n_features,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
    )
    module = _FakeModule()
    residual = torch.randn(1, 3, d_model, dtype=torch.float64)
    selected_indices = (1, 3)
    scales = (2.0, 0.5)
    beta = (-1.0, -0.5)
    last_token = residual[0, 2]
    expected_intervened = apply_additive_jumprelu_sae_delta(
        residual=last_token,
        selected_indices=selected_indices,
        scales=scales,
        beta=beta,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
    )
    assert not torch.equal(expected_intervened, last_token)

    with _patch_resolver({layer: module}):
        with install_multi_layer_hooks(
            model=SimpleNamespace(),
            resolver_id="gemma3_resid_post",
            saes={layer: sae},
            selected_keys=((layer, 1), (layer, 3)),
            scales=list(scales),
            beta=list(beta),
            token_scope="last_prompt_token",
            prompt_lengths=[3],
            k=None,
        ):
            out = module.run(residual.clone())

    assert torch.allclose(out[0, 2], expected_intervened, atol=1e-12, rtol=0.0)
    assert torch.equal(out[0, 0], residual[0, 0])
    assert torch.equal(out[0, 1], residual[0, 1])


@pytest.mark.unit
def test_hooks__beta_zero__invokes_delta_fn_on_masked_tokens() -> None:
    """DEC-086: β=0 must still call delta_fn (no all_zero short-circuit)."""
    torch.manual_seed(2)
    d_model = 4
    n_features = 3
    layer = 9
    encoder_weight = torch.randn(n_features, d_model, dtype=torch.float64)
    encoder_bias = torch.randn(n_features, dtype=torch.float64)
    threshold = torch.full((n_features,), 0.05, dtype=torch.float64)
    decoder_weight = torch.randn(n_features, d_model, dtype=torch.float64)
    sae = _fake_sae(
        d_model=d_model,
        n_features=n_features,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        threshold=threshold,
        decoder_weight=decoder_weight,
    )
    module = _FakeModule()
    residual = torch.randn(2, 4, d_model, dtype=torch.float64)
    prompt_lengths = [3, 2]
    call_count = {"n": 0}

    def recording_delta_fn(
        *,
        residual: torch.Tensor,
        selected_indices: Any,
        scales: Any,
        beta: Any,
        sae: Any,
    ) -> torch.Tensor:
        del selected_indices, scales, beta, sae
        call_count["n"] += 1
        return residual

    with _patch_resolver({layer: module}):
        with install_multi_layer_hooks(
            model=SimpleNamespace(),
            resolver_id="gemma3_resid_post",
            saes={layer: sae},
            selected_keys=((layer, 0),),
            scales=[1.0],
            beta=[0.0],
            token_scope="last_prompt_token",
            prompt_lengths=prompt_lengths,
            k=None,
            delta_fn=recording_delta_fn,
        ):
            out = module.run(residual.clone())

    # last_prompt_token → one masked position per batch row
    assert call_count["n"] == 2
    assert torch.equal(out, residual)
