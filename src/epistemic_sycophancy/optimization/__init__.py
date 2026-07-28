"""Optimizer package (Phase H)."""

from epistemic_sycophancy.optimization.cmaes import CMAESOptimizer
from epistemic_sycophancy.optimization.objective import evaluate_optimizer_objective
from epistemic_sycophancy.optimization.projected_adam import ProjectedAdam

__all__ = ["CMAESOptimizer", "ProjectedAdam", "evaluate_optimizer_objective"]
