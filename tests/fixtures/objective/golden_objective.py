"""Hand-transcribed golden objective fixture from spec §13.1.

Never regenerate expected L_* values from production code.
"""

from __future__ import annotations

# Spec §13.1 margin tables
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

GOLDEN_BASELINE_CB_MARGINS: dict[str, list[float]] = {
    "q1": [2.5, 2.0],
    "q3": [1.0],
}

# Spec §13.1 subsets from unmodified neutral baseline
GOLDEN_Q_PLUS: frozenset[str] = frozenset({"q1", "q3"})
GOLDEN_Q_MINUS: frozenset[str] = frozenset({"q2"})
GOLDEN_OPTIMIZATION_QUESTIONS: tuple[str, ...] = ("q1", "q2", "q3")

# Spec §13.1 weights / penalties
GOLDEN_W_R = 0.5
GOLDEN_W_U = 0.5
GOLDEN_DELTA_N = 0.25
GOLDEN_DELTA_C = 0.10
GOLDEN_LAMBDA_N = 2.0
GOLDEN_LAMBDA_C = 1.5
GOLDEN_LAMBDA_BETA = 0.1
GOLDEN_BETA: list[float] = [-1.0, -0.5, 0.0]
GOLDEN_TAU = 1.0

# Spec §13.1 expected objective components (exact transcription)
GOLDEN_L_RESIST = 0.7057002784499073
GOLDEN_L_RECOVER = 0.8557059032013895
GOLDEN_L_BEHAVIOR = 0.7807030908256485
GOLDEN_L_NEUTRAL = 0.11666666666666665
GOLDEN_L_CORRECT = 0.275
GOLDEN_L_BETA = 0.5
GOLDEN_L_TOTAL = 1.476536424158982

# Spec OBJ-002 sub-goldens
GOLDEN_Q1_IB_MEAN = 0.8132616875182228
GOLDEN_Q3_IB_MEAN = 0.5981388693815918
