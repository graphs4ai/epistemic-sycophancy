"""StudyConfig: stack + experiment + run (Phase L CFGFILE-001 / DEC-056)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from epistemic_sycophancy.config.schema import (
    ExperimentConfig,
    InvalidExperimentConfig,
)
from epistemic_sycophancy.stack.config import ExperimentStackConfig

_ALLOWED_SMOKE_SPLITS = frozenset({"feature_selection", "optimization"})
_ALLOWED_OPTIMIZER_KINDS = frozenset({"projected_adam", "cmaes"})
_ALLOWED_BUDGET_MATCH_ON = frozenset({"n_objective_evals", "n_forward_equiv"})
_ALLOWED_ORDER_REGIMES = frozenset({"CF", "IF", "RO"})


@dataclass(frozen=True)
class StudySmokeConfig:
    """Tiny deterministic corpus subset for Phase L smokes (DEC-059)."""

    n_questions: int | None = None
    split: str | None = None
    seed: int | None = None
    question_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        has_allowlist = self.question_ids is not None
        has_n = self.n_questions is not None
        if has_allowlist == has_n:
            # XOR: exactly one of allowlist or {n, split, seed}
            if not has_allowlist:
                raise InvalidExperimentConfig(
                    "run.smoke requires either question_ids or "
                    "{n_questions, split, seed}; got neither"
                )
            raise InvalidExperimentConfig(
                "run.smoke requires either question_ids or "
                "{n_questions, split, seed}; got both"
            )
        if has_allowlist:
            ids = tuple(str(q) for q in self.question_ids or ())
            if not ids:
                raise InvalidExperimentConfig(
                    "run.smoke.question_ids must be a nonempty sequence when set"
                )
            object.__setattr__(self, "question_ids", ids)
            return
        if self.n_questions is None or self.n_questions < 1:
            raise InvalidExperimentConfig(
                f"run.smoke.n_questions must be a positive int; got {self.n_questions!r}"
            )
        if self.split not in _ALLOWED_SMOKE_SPLITS:
            raise InvalidExperimentConfig(
                "run.smoke.split must be one of "
                f"{sorted(_ALLOWED_SMOKE_SPLITS)}; got {self.split!r}"
            )
        if self.seed is None or not isinstance(self.seed, int) or isinstance(
            self.seed, bool
        ):
            raise InvalidExperimentConfig(
                f"run.smoke.seed must be an explicit int; got {self.seed!r}"
            )


@dataclass(frozen=True)
class StudyOptimizerConfig:
    """Optimizer knobs for opt_smoke / study (DEC-062); no silent defaults."""

    kind: str
    max_steps: int
    adam_lr: float | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_eps: float | None = None
    adam_microbatch_questions: int | None = None
    cma_seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_OPTIMIZER_KINDS:
            raise InvalidExperimentConfig(
                f"run.optimizer.kind must be one of {sorted(_ALLOWED_OPTIMIZER_KINDS)}; "
                f"got {self.kind!r}"
            )
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise InvalidExperimentConfig(
                f"run.optimizer.max_steps must be a positive int; got {self.max_steps!r}"
            )
        if self.max_steps < 1:
            raise InvalidExperimentConfig(
                f"run.optimizer.max_steps must be a positive int; got {self.max_steps!r}"
            )
        if self.kind == "projected_adam":
            for name, value in (
                ("adam_lr", self.adam_lr),
                ("adam_beta1", self.adam_beta1),
                ("adam_beta2", self.adam_beta2),
                ("adam_eps", self.adam_eps),
                ("adam_microbatch_questions", self.adam_microbatch_questions),
            ):
                if value is None:
                    raise InvalidExperimentConfig(
                        f"run.optimizer.{name} must be explicit for projected_adam"
                    )
            if (
                not isinstance(self.adam_microbatch_questions, int)
                or isinstance(self.adam_microbatch_questions, bool)
                or self.adam_microbatch_questions < 1
            ):
                raise InvalidExperimentConfig(
                    "run.optimizer.adam_microbatch_questions must be a positive int; "
                    f"got {self.adam_microbatch_questions!r}"
                )
        elif self.kind == "cmaes":
            if self.cma_seed is None or not isinstance(self.cma_seed, int) or isinstance(
                self.cma_seed, bool
            ):
                raise InvalidExperimentConfig(
                    f"run.optimizer.cma_seed must be an explicit int; got {self.cma_seed!r}"
                )


@dataclass(frozen=True)
class StudyOptimizeConfig:
    """Non-smoke optimize budgets (DEC-066 / DEC-068); distinct from smoke max_steps."""

    budget_match_on: str
    max_steps: int | None = None
    n_trials: int | None = None
    population_size: int | None = None
    n_questions: int | None = None
    question_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.budget_match_on not in _ALLOWED_BUDGET_MATCH_ON:
            raise InvalidExperimentConfig(
                "run.optimize.budget_match_on must be one of "
                f"{sorted(_ALLOWED_BUDGET_MATCH_ON)}; got {self.budget_match_on!r}"
            )
        has_adam = self.max_steps is not None
        has_cma = self.n_trials is not None or self.population_size is not None
        if has_adam == has_cma:
            # Exactly one of Adam max_steps or CMA (n_trials+population_size).
            if not has_adam and not has_cma:
                raise InvalidExperimentConfig(
                    "run.optimize requires max_steps (Adam) or "
                    "n_trials+population_size (CMA); got neither"
                )
            raise InvalidExperimentConfig(
                "run.optimize requires either max_steps (Adam) or "
                "n_trials+population_size (CMA); got both"
            )
        if has_adam:
            if (
                not isinstance(self.max_steps, int)
                or isinstance(self.max_steps, bool)
                or self.max_steps < 1
            ):
                raise InvalidExperimentConfig(
                    f"run.optimize.max_steps must be a positive int; got {self.max_steps!r}"
                )
        else:
            if (
                not isinstance(self.n_trials, int)
                or isinstance(self.n_trials, bool)
                or self.n_trials < 1
            ):
                raise InvalidExperimentConfig(
                    f"run.optimize.n_trials must be a positive int; got {self.n_trials!r}"
                )
            if (
                not isinstance(self.population_size, int)
                or isinstance(self.population_size, bool)
                or self.population_size < 1
            ):
                raise InvalidExperimentConfig(
                    "run.optimize.population_size must be a positive int; "
                    f"got {self.population_size!r}"
                )

        has_allowlist = self.question_ids is not None
        has_n = self.n_questions is not None
        if has_allowlist and has_n:
            raise InvalidExperimentConfig(
                "run.optimize requires either question_ids or n_questions, not both"
            )
        if has_allowlist:
            ids = tuple(str(q) for q in self.question_ids or ())
            if not ids:
                raise InvalidExperimentConfig(
                    "run.optimize.question_ids must be a nonempty sequence when set"
                )
            object.__setattr__(self, "question_ids", ids)
        if has_n:
            if (
                not isinstance(self.n_questions, int)
                or isinstance(self.n_questions, bool)
                or self.n_questions < 1
            ):
                raise InvalidExperimentConfig(
                    "run.optimize.n_questions must be a positive int; "
                    f"got {self.n_questions!r}"
                )


@dataclass(frozen=True)
class StudyRunConfig:
    """Stage / smoke / optimizer / optimize run options (DEC-056 / DEC-087)."""

    artifact_dir: str
    order_regime: str
    feature_chunk_size: int
    prompt_batch_size: int
    smoke: StudySmokeConfig
    optimizer: StudyOptimizerConfig
    optimize: StudyOptimizeConfig

    def __post_init__(self) -> None:
        if self.artifact_dir is None or not str(self.artifact_dir).strip():
            raise InvalidExperimentConfig(
                f"run.artifact_dir must be an explicit non-empty string; "
                f"got {self.artifact_dir!r}"
            )
        regime = str(self.order_regime).upper()
        if regime not in _ALLOWED_ORDER_REGIMES:
            raise InvalidExperimentConfig(
                "run.order_regime must be one of {'CF', 'IF', 'RO'}; "
                f"got {self.order_regime!r}"
            )
        object.__setattr__(self, "order_regime", regime)
        if (
            not isinstance(self.feature_chunk_size, int)
            or isinstance(self.feature_chunk_size, bool)
            or self.feature_chunk_size < 1
        ):
            raise InvalidExperimentConfig(
                "run.feature_chunk_size must be a positive int; "
                f"got {self.feature_chunk_size!r}"
            )
        if (
            not isinstance(self.prompt_batch_size, int)
            or isinstance(self.prompt_batch_size, bool)
            or self.prompt_batch_size < 1
        ):
            raise InvalidExperimentConfig(
                "run.prompt_batch_size must be a positive int; "
                f"got {self.prompt_batch_size!r}"
            )
        if self.smoke is None:
            raise InvalidExperimentConfig("run.smoke must be explicit")
        if self.optimizer is None:
            raise InvalidExperimentConfig("run.optimizer must be explicit")
        if self.optimize is None:
            raise InvalidExperimentConfig("run.optimize must be explicit")


@dataclass(frozen=True)
class StudyConfig:
    """Validated study: stack + experiment + run (DEC-056)."""

    stack: ExperimentStackConfig
    experiment: ExperimentConfig
    run: StudyRunConfig

    def __post_init__(self) -> None:
        if self.stack is None:
            raise InvalidExperimentConfig("stack must be explicit")
        if self.experiment is None:
            raise InvalidExperimentConfig("experiment must be explicit")
        if self.run is None:
            raise InvalidExperimentConfig("run must be explicit")


def build_study_config(
    *,
    stack: ExperimentStackConfig,
    experiment: ExperimentConfig,
    run: StudyRunConfig,
) -> StudyConfig:
    """Factory mirroring the validated StudyConfig constructor."""
    return StudyConfig(stack=stack, experiment=experiment, run=run)


def study_order_regime(study: StudyConfig) -> str:
    """Return the single answer-order regime for this study (DEC-087)."""
    return str(study.run.order_regime)


def coerce_order_regimes(values: Sequence[str]) -> tuple[str, ...]:
    """Deprecated helper retained for call-site migration; prefer order_regime."""
    return tuple(str(v) for v in values)
