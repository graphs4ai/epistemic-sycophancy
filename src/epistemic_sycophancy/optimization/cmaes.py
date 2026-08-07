"""CMA-ES coefficient optimizer wrapper (Phase H; DEC-030)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import cma
import numpy as np

from epistemic_sycophancy.prompts.ordering import hash_ro_manifest


class CMAESOptimizer:
    """CMA-ES over β with full-corpus objective evaluations (OPT-002)."""

    def __init__(
        self,
        *,
        x0: Sequence[float],
        sigma0: float,
        cma_seed: int,
        beta_lower: float,
        beta_upper: float,
        eligible_question_ids: Sequence[str],
        ro_manifest: Mapping[str, object] | None = None,
    ) -> None:
        if cma_seed is None:
            raise ValueError("cma_seed is required (DEC-030)")
        if not (beta_lower <= beta_upper):
            raise ValueError(
                "bounds require beta_lower <= beta_upper; "
                f"got beta_lower={beta_lower!r}, beta_upper={beta_upper!r}"
            )
        self.beta_lower = float(beta_lower)
        self.beta_upper = float(beta_upper)
        self.eligible_question_ids = tuple(eligible_question_ids)
        # Store the caller-provided frozen manifest; never resample (OPT-004).
        self.ro_manifest = ro_manifest
        self._ro_manifest_hash = (
            hash_ro_manifest(dict(ro_manifest)) if ro_manifest is not None else None
        )
        self._es = cma.CMAEvolutionStrategy(
            list(x0),
            float(sigma0),
            {
                "seed": int(cma_seed),
                "verbose": -9,
                "bounds": [self.beta_lower, self.beta_upper],
            },
        )

    @property
    def ro_manifest_hash(self) -> str:
        """Frozen RO manifest hash reused across trials (OPT-004 / DEC-009)."""
        if self._ro_manifest_hash is None:
            raise ValueError("ro_manifest was not provided to CMAESOptimizer")
        return self._ro_manifest_hash

    def ask(self) -> list[list[float]]:
        """Propose candidate β vectors (clamped to configured bounds)."""
        raw = self._es.ask()
        return [self._clamp_beta(list(map(float, candidate))) for candidate in raw]

    def evaluate_candidate(
        self,
        beta: Sequence[float],
        *,
        evaluate_on_questions: Callable[[list[float], list[str]], float],
    ) -> float:
        """Evaluate one candidate on the full eligible optimization corpus."""
        return float(
            evaluate_on_questions(
                list(map(float, beta)),
                list(self.eligible_question_ids),
            )
        )

    def tell(self, solutions: Sequence[Sequence[float]], values: Sequence[float]) -> None:
        """Update CMA-ES from evaluated candidates."""
        self._es.tell(
            [np.asarray(s, dtype=np.float64) for s in solutions],
            list(map(float, values)),
        )

    def _clamp_beta(self, beta: list[float]) -> list[float]:
        return [
            min(self.beta_upper, max(self.beta_lower, float(v))) for v in beta
        ]
