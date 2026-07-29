"""Experiment stack configuration (Phase K)."""

from __future__ import annotations

from dataclasses import dataclass

from epistemic_sycophancy.models.spec import ModelSpec
from epistemic_sycophancy.sae.spec import SaeSiteSpec


@dataclass(frozen=True)
class HookSpec:
    """Token-scope and resid_post module-resolver identity (DEC-015)."""

    token_scope: str
    resolver_id: str
    k: int | None = None

    def __post_init__(self) -> None:
        if self.token_scope is None or not str(self.token_scope):
            raise ValueError("token_scope must be explicit (DEC-015)")
        if not self.resolver_id:
            raise ValueError("resolver_id must be a non-empty string")
        if self.token_scope == "last_k_prompt_tokens":
            if self.k is None or self.k < 1:
                raise ValueError(
                    "last_k_prompt_tokens requires positive int k; "
                    f"got k={self.k!r}"
                )


@dataclass(frozen=True)
class ExperimentStackConfig:
    """Config-driven model + SAE site + hook policy for InterventionStack."""

    model: ModelSpec
    sae: SaeSiteSpec
    hooks: HookSpec
