"""Feature-selection score artifacts (Phase F, DEC-024)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError


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
    question_ids: frozenset[str]
    component: str | None = None
    order_regime: str | None = None
    model_revision_hash: str | None = None
    sae_revision_hash: str | None = None
    scope: str | None = None
    scale_source: str | None = None
    dataset_manifest_hash: str | None = None
    fingerprint: str | None = None

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, object]] | Sequence[FeatureSelectionRow],
        question_ids: frozenset[str] | None = None,
        feature_selection_question_ids: frozenset[str] | None = None,
        component: str | None = None,
        order_regime: str | None = None,
        model_revision_hash: str | None = None,
        sae_revision_hash: str | None = None,
        scope: str | None = None,
        scale_source: str | None = None,
        dataset_manifest_hash: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        if rows and isinstance(rows[0], FeatureSelectionRow):
            parsed = tuple(rows)  # type: ignore[arg-type]
        else:
            parsed = tuple(
                FeatureSelectionRow(
                    layer=int(row["layer"]),  # type: ignore[index, arg-type]
                    feature_id=int(row["feature_id"]),  # type: ignore[index, arg-type]
                    signed_jacobian=float(row["signed_jacobian"]),  # type: ignore[index, arg-type]
                    absolute_sensitivity=float(
                        row["absolute_sensitivity"]  # type: ignore[index, arg-type]
                    ),
                    raw_projection=float(row["raw_projection"]),  # type: ignore[index, arg-type]
                    mean_active_rate=float(row["mean_active_rate"]),  # type: ignore[index, arg-type]
                    feature_scale=float(row["feature_scale"]),  # type: ignore[index, arg-type]
                    suppression_beneficial=bool(
                        row["suppression_beneficial"]  # type: ignore[index]
                    ),
                    preferred_bidirectional_sign=float(
                        row["preferred_bidirectional_sign"]  # type: ignore[index, arg-type]
                    ),
                    n_questions=int(row["n_questions"]),  # type: ignore[index, arg-type]
                    n_prompts=int(row["n_prompts"]),  # type: ignore[index, arg-type]
                )
                for row in rows  # type: ignore[union-attr]
            )
        ids = frozenset() if question_ids is None else frozenset(question_ids)
        if feature_selection_question_ids is not None:
            leaked = ids - frozenset(feature_selection_question_ids)
            if leaked:
                raise HoldoutAccessError(
                    "feature selection may only use feature_selection-split "
                    f"question IDs; leaked={sorted(leaked)!r}"
                )
        object.__setattr__(self, "rows", parsed)
        object.__setattr__(self, "question_ids", ids)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "order_regime", order_regime)
        object.__setattr__(self, "model_revision_hash", model_revision_hash)
        object.__setattr__(self, "sae_revision_hash", sae_revision_hash)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "scale_source", scale_source)
        object.__setattr__(self, "dataset_manifest_hash", dataset_manifest_hash)
        object.__setattr__(self, "fingerprint", fingerprint)


def freeze_feature_selection_artifact(
    *,
    artifact: FeatureSelectionArtifact,
    component: str,
    order_regime: str,
    model_revision_hash: str,
    sae_revision_hash: str,
    scope: str,
    scale_source: str,
    dataset_manifest_hash: str,
) -> FeatureSelectionArtifact:
    """Attach reproducibility hashes and a deterministic fingerprint (FEAT-029)."""
    row_material = ";".join(
        f"{row.layer},{row.feature_id},{row.signed_jacobian},"
        f"{row.absolute_sensitivity},{row.raw_projection},"
        f"{row.mean_active_rate},{row.feature_scale}"
        for row in artifact.rows
    )
    material = "|".join(
        [
            component,
            order_regime,
            model_revision_hash,
            sae_revision_hash,
            scope,
            scale_source,
            dataset_manifest_hash,
            ",".join(sorted(artifact.question_ids)),
            row_material,
        ]
    )
    fingerprint = sha256(material.encode("utf-8")).hexdigest()
    return FeatureSelectionArtifact(
        rows=artifact.rows,
        question_ids=artifact.question_ids,
        component=component,
        order_regime=order_regime,
        model_revision_hash=model_revision_hash,
        sae_revision_hash=sae_revision_hash,
        scope=scope,
        scale_source=scale_source,
        dataset_manifest_hash=dataset_manifest_hash,
        fingerprint=fingerprint,
    )
