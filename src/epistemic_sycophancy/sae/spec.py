"""SAE site specs for configurable multi-layer stacks (Phase K)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class InvalidSaeSiteSpec(Exception):
    """Raised when an SAE site specification violates a required invariant."""


# Known GemmaScope2 4B-IT resid_post subset layers (25/50/65/85% depth).
GEMMASCOPE2_4B_IT_RESID_POST_SUBSET_LAYERS: frozenset[int] = frozenset({9, 17, 22, 29})


@dataclass(frozen=True)
class SaeSiteSpec:
    """Pinned SAE release, site, width, L0, and layer subset."""

    release: str
    site: str
    width: str
    l0: str
    layers: tuple[int, ...]

    def __init__(
        self,
        *,
        release: str,
        site: str,
        width: str | None,
        l0: str | None,
        layers: Sequence[int],
    ) -> None:
        if width is None or not str(width):
            raise InvalidSaeSiteSpec(
                f"width must be an explicit non-empty string; got {width!r}"
            )
        if l0 is None or not str(l0):
            raise InvalidSaeSiteSpec(
                f"l0 must be an explicit non-empty string; got {l0!r}"
            )
        layer_tuple = tuple(int(layer) for layer in layers)
        if not layer_tuple:
            raise InvalidSaeSiteSpec("layers must be a nonempty sequence")
        if len(layer_tuple) != len(set(layer_tuple)):
            raise InvalidSaeSiteSpec(
                f"layers must be unique; got {layer_tuple!r}"
            )
        unknown = set(layer_tuple) - GEMMASCOPE2_4B_IT_RESID_POST_SUBSET_LAYERS
        if unknown:
            raise InvalidSaeSiteSpec(
                "layers must be a subset of "
                f"{sorted(GEMMASCOPE2_4B_IT_RESID_POST_SUBSET_LAYERS)}; "
                f"unknown={sorted(unknown)}"
            )
        if not release:
            raise InvalidSaeSiteSpec("release must be a non-empty string")
        if not site:
            raise InvalidSaeSiteSpec("site must be a non-empty string")
        object.__setattr__(self, "release", release)
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "width", str(width))
        object.__setattr__(self, "l0", str(l0))
        object.__setattr__(self, "layers", layer_tuple)
