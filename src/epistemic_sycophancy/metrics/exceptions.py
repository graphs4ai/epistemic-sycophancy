"""Metric and baseline-partition exceptions."""


class DegenerateBaselineError(Exception):
    """Raised when a required baseline subset is empty after partitioning."""
