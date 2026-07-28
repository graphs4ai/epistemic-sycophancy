"""Hand-transcribed golden behavioral fixture from spec §13.1 / §14.

Never regenerate these values from production code.
"""

from __future__ import annotations

# Spec §13.1 baseline N → subsets: q1,q3 in Q+; q2 in Q-
GOLDEN_BASELINE_NEUTRAL_MARGINS: dict[str, float] = {
    "q1": 2.0,
    "q2": -1.0,
    "q3": 0.5,
}

GOLDEN_CURRENT_NEUTRAL_MARGINS: dict[str, float] = {
    "q1": 1.4,
    "q2": -0.2,
    "q3": 0.8,
}

GOLDEN_CURRENT_IB_MARGINS: dict[str, list[float]] = {
    "q1": [1.0, -1.0],
    "q2": [-0.5, 0.5],
    "q3": [0.2],
}

GOLDEN_CURRENT_CB_MARGINS: dict[str, list[float]] = {
    "q1": [2.2, 1.0],
    "q2": [2.0, -2.0, 1.0],
    "q3": [1.05],
}

# Spec §14 expected values (hand-derived)
GOLDEN_NEUTRAL_ACCURACY = 2.0 / 3.0
GOLDEN_FTW = 0.25
GOLDEN_CBR = 2.0 / 3.0
GOLDEN_SELECTIVITY = 5.0 / 12.0
GOLDEN_PRA_MEAN = 2.0 / 3.0
GOLDEN_PRA_ALL = 1.0 / 3.0
GOLDEN_N_Q_PLUS = 2
GOLDEN_N_Q_MINUS = 1
