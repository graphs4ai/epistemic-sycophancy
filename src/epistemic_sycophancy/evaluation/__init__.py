"""Cross-order evaluation package."""

from epistemic_sycophancy.evaluation.cross_order import (
    CrossOrderCellRecord,
    build_cross_order_matrix,
)
from epistemic_sycophancy.evaluation.toy_e2e import (
    ToyE2EBaselineResult,
    build_dec046_corpus,
    run_toy_e2e_baseline,
)

__all__ = [
    "CrossOrderCellRecord",
    "ToyE2EBaselineResult",
    "build_cross_order_matrix",
    "build_dec046_corpus",
    "run_toy_e2e_baseline",
]
