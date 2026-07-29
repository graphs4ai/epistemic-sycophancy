"""Load GemmaScope2 SAEs via sae-lens (Phase K RUN-003/004)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from epistemic_sycophancy.sae.spec import SaeSiteSpec


@dataclass(frozen=True)
class SaeHandle:
    """Frozen handle for one layer's loaded SAE."""

    layer: int
    release: str
    sae_id: str
    d_in: int
    d_sae: int
    decoder_width: int
    decoder_weight: torch.Tensor
    sae: Any


def sae_id_for_layer(spec: SaeSiteSpec, layer: int) -> str:
    """Build DEC-051 sae_id: layer_{L}_width_65k_l0_medium (width/l0 from spec)."""
    if layer not in spec.layers:
        raise ValueError(
            f"layer {layer} is not in SaeSiteSpec.layers={spec.layers!r}"
        )
    # Spec stores canonical tokens like "width_65k" and "l0_medium".
    return f"layer_{layer}_{spec.width}_{spec.l0}"


def load_sae(
    *,
    spec: SaeSiteSpec,
    layer: int,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> SaeHandle:
    """Load one JumpReLU SAE for ``layer`` from the pinned release (DEC-051)."""
    from sae_lens import SAE

    sae_id = sae_id_for_layer(spec, layer)
    sae = SAE.from_pretrained(
        release=spec.release,
        sae_id=sae_id,
        device=device,
        dtype=dtype,
    )
    d_in = int(sae.cfg.d_in)
    d_sae = int(sae.cfg.d_sae)
    decoder = sae.W_dec.detach()
    if tuple(decoder.shape) != (d_sae, d_in):
        raise ValueError(
            f"unexpected decoder shape {tuple(decoder.shape)}; "
            f"expected {(d_sae, d_in)}"
        )
    return SaeHandle(
        layer=layer,
        release=spec.release,
        sae_id=sae_id,
        d_in=d_in,
        d_sae=d_sae,
        decoder_width=d_in,
        decoder_weight=decoder,
        sae=sae,
    )
