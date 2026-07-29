"""InterventionStack: model + multi-layer SAEs + hooks (Phase K)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from epistemic_sycophancy.models.load import LoadedModel, load_model
from epistemic_sycophancy.sae.load import SaeHandle, load_sae_stack
from epistemic_sycophancy.stack.config import ExperimentStackConfig
from epistemic_sycophancy.stack.hooks import install_multi_layer_hooks
from epistemic_sycophancy.stack.resolver import resolve_resid_post_module


@dataclass(frozen=True)
class InterventionStack:
    """Frozen-in-memory model, tokenizer, and per-layer SAEs."""

    config: ExperimentStackConfig
    loaded: LoadedModel
    saes: Mapping[int, SaeHandle]

    @property
    def model(self) -> Any:
        return self.loaded.model

    @property
    def tokenizer(self) -> Any:
        return self.loaded.tokenizer

    @property
    def device(self) -> torch.device:
        return self.loaded.device

    def _language_model(self) -> Any:
        return self.model.model.language_model

    def _encode_texts(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def capture_layer_residuals(
        self,
        *,
        texts: Sequence[str],
        layers: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """Run a text-only language_model forward and capture resid_post tensors."""
        encoded = self._encode_texts(texts)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer: int):
            def hook(module: Any, inputs: Any, output: Any) -> Any:
                del module, inputs
                tensor = output[0] if isinstance(output, tuple) else output
                captured[layer] = tensor.detach().clone()
                return output

            return hook

        for layer in layers:
            module = resolve_resid_post_module(
                self.model,
                layer=layer,
                resolver_id=self.config.hooks.resolver_id,
            )
            handles.append(module.register_forward_hook(make_hook(layer)))
        try:
            with torch.no_grad():
                self._language_model()(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                )
        finally:
            for handle in handles:
                handle.remove()
        missing = set(layers) - set(captured)
        if missing:
            raise RuntimeError(f"failed to capture residuals for layers {sorted(missing)}")
        return captured

    @contextmanager
    def install_hooks(
        self,
        *,
        selected_keys: Sequence[tuple[int, int]],
        scales: Sequence[float],
        beta: Sequence[float],
        prompt_lengths: Sequence[int],
    ) -> Iterator[None]:
        """Install simultaneous multi-layer hooks for the given β layout.

        ``prompt_lengths`` must be the per-batch prompt token counts (DEC-015);
        the former ``[1]`` stub is forbidden for nonzero-β token scope.
        """
        if not prompt_lengths:
            raise ValueError("prompt_lengths must be a nonempty sequence")
        with install_multi_layer_hooks(
            model=self.model,
            resolver_id=self.config.hooks.resolver_id,
            saes=self.saes,
            selected_keys=selected_keys,
            scales=scales,
            beta=beta,
            token_scope=self.config.hooks.token_scope,
            prompt_lengths=prompt_lengths,
            k=self.config.hooks.k,
        ):
            yield


def load_stack(
    cfg: ExperimentStackConfig,
    *,
    _load_model: Any | None = None,
    _load_sae_stack: Any | None = None,
) -> InterventionStack:
    """Load model + SAE stack from config (config-only layer selection)."""
    load_model_fn = _load_model or load_model
    load_sae_stack_fn = _load_sae_stack or load_sae_stack
    loaded = load_model_fn(cfg.model)
    device = "cuda" if loaded.device.type == "cuda" else "cpu"
    saes = load_sae_stack_fn(
        spec=cfg.sae,
        device=device,
        dtype=cfg.model.dtype,
    )
    return InterventionStack(config=cfg, loaded=loaded, saes=saes)
