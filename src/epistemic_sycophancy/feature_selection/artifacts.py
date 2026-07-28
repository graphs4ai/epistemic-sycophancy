"""Feature-selection score artifacts (Phase F, DEC-024)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSelectionRow:
    """One logged feature-selection score row (FEAT-027 / DEC-024)."""

    layer: int
    feature_id: int
    signed_jacobian: float
    absolute_sensitivity: float
    raw_projection: float
    mean_active_rate: float
    feature_scale: float
    suppression_beneficial: bool
    preferred_bidirectional_sign: float
    n_questions: int
    n_prompts: int


@dataclass(frozen=True)
class FeatureSelectionArtifact:
    """Logged feature-selection scores for one component × order (DEC-024)."""

    rows: tuple[FeatureSelectionRow, ...]

    def __init__(self, *, rows: Sequence[Mapping[str, object]]) -> None:
        parsed = tuple(
            FeatureSelectionRow(
                layer=int(row["layer"]),  # type: ignore[arg-type]
                feature_id=int(row["feature_id"]),  # type: ignore[arg-type]
                signed_jacobian=float(row["signed_jacobian"]),  # type: ignore[arg-type]
                absolute_sensitivity=float(row["absolute_sensitivity"]),  # type: ignore[arg-type]
                raw_projection=float(row["raw_projection"]),  # type: ignore[arg-type]
                mean_active_rate=float(row["mean_active_rate"]),  # type: ignore[arg-type]
                feature_scale=float(row["feature_scale"]),  # type: ignore[arg-type]
                suppression_beneficial=bool(row["suppression_beneficial"]),
                preferred_bidirectional_sign=float(
                    row["preferred_bidirectional_sign"]  # type: ignore[arg-type]
                ),
                n_questions=int(row["n_questions"]),  # type: ignore[arg-type]
                n_prompts=int(row["n_prompts"]),  # type: ignore[arg-type]
            )
            for row in rows
        )
        object.__setattr__(self, "rows", parsed)
