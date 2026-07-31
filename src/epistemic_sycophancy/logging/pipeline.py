"""Operational pipeline logging for staged experiment runs (DEC-089)."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from typing import Any

PIPELINE_LOGGER_NAME = "epistemic_sycophancy.pipeline"

_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def get_pipeline_logger() -> logging.Logger:
    """Return the dedicated pipeline logger (does not configure handlers)."""
    return logging.getLogger(PIPELINE_LOGGER_NAME)


def configure_pipeline_logging(level: str = "INFO") -> logging.Logger:
    """Attach a single stderr StreamHandler at ``level`` to the pipeline logger.

    Idempotent for the same level: replaces prior pipeline StreamHandlers so
    repeated CLI invocations in-process do not stack handlers.
    """
    name = str(level).upper()
    if name not in _LEVEL_NAMES:
        raise ValueError(
            f"unsupported log level {level!r}; expected one of {sorted(_LEVEL_NAMES)}"
        )
    numeric = _LEVEL_NAMES[name]
    logger = get_pipeline_logger()
    logger.setLevel(numeric)
    # Drop prior stderr handlers we own so configure is idempotent.
    kept: list[logging.Handler] = []
    for handler in list(logger.handlers):
        if getattr(handler, "_epistemic_pipeline_handler", False):
            logger.removeHandler(handler)
            handler.close()
        else:
            kept.append(handler)
    del kept
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(numeric)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler._epistemic_pipeline_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_stage_start(stage: str, *, study_fp: str = "", extra: Mapping[str, Any] | None = None) -> float:
    """Emit stage-start INFO; return monotonic start time for duration pairing."""
    logger = get_pipeline_logger()
    parts = [f"stage={stage} starting"]
    if study_fp:
        parts.append(f"study_fp={study_fp[:12]}…")
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))
    return time.monotonic()


def log_stage_end(
    stage: str,
    *,
    ok: bool,
    message: str,
    started_at: float,
    artifacts: Mapping[str, str] | None = None,
) -> None:
    """Emit stage-end INFO/ERROR with wall duration and optional artifact paths."""
    logger = get_pipeline_logger()
    elapsed = time.monotonic() - started_at
    level = logging.INFO if ok else logging.ERROR
    parts = [
        f"stage={stage} {'completed' if ok else 'FAILED'}",
        f"ok={ok}",
        f"elapsed_s={elapsed:.3f}",
        f"message={message}",
    ]
    if artifacts:
        for name, path in artifacts.items():
            parts.append(f"artifact.{name}={path}")
    logger.log(level, " | ".join(parts))


def log_progress(event: str, **fields: Any) -> None:
    """Emit a structured INFO progress line (optimize step, FS component, …)."""
    logger = get_pipeline_logger()
    parts = [f"progress={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))


def log_audit(event: str, **fields: Any) -> None:
    """Emit WARNING for high-consequence audit events (freeze seal, holdout unseal)."""
    logger = get_pipeline_logger()
    parts = [f"audit={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.warning(" | ".join(parts))
