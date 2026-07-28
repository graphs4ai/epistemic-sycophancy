"""Scoring-domain exceptions."""

from __future__ import annotations


class InvalidScoreError(Exception):
    """Raised when a candidate score is missing or non-finite (DEC-012)."""
