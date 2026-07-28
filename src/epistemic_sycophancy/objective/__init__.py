"""Objective losses and question-macro aggregation."""

from epistemic_sycophancy.objective.aggregation import question_macro_mean
from epistemic_sycophancy.objective.losses import (
    logistic_margin_loss,
    mean_logistic_margin_loss,
)

__all__ = [
    "logistic_margin_loss",
    "mean_logistic_margin_loss",
    "question_macro_mean",
]
