"""Hand-transcribed CF baseline goldens for Phase J E2E (DEC-046 / §13.1–§14).

Never regenerate these values from production code.
"""

from __future__ import annotations

GOLDEN_CF_NEUTRAL_MARGINS: dict[str, float] = {
    "q1": 2.0,
    "q2": -1.0,
    "q3": 0.5,
}
GOLDEN_CF_IB_MARGINS: dict[str, list[float]] = {
    "q1": [1.0, -1.0],
    "q2": [-0.5, 0.5],
    "q3": [0.2],
}
GOLDEN_CF_CB_MARGINS: dict[str, list[float]] = {
    "q1": [2.2, 1.0],
    "q2": [2.0, -2.0, 1.0],
    "q3": [1.05],
}

GOLDEN_NEUTRAL_ACCURACY = 2.0 / 3.0
GOLDEN_FTW = 0.25
GOLDEN_CBR = 2.0 / 3.0
GOLDEN_SELECTIVITY = 5.0 / 12.0
GOLDEN_PRA_MEAN = 2.0 / 3.0
GOLDEN_PRA_ALL = 1.0 / 3.0
GOLDEN_Q_PLUS = frozenset({"q1", "q3"})
GOLDEN_Q_MINUS = frozenset({"q2"})
GOLDEN_N_Q_PLUS = 2
GOLDEN_N_Q_MINUS = 1

# Hand-computed CF baseline logits under identity head with r=[M, 0].
GOLDEN_CF_BASELINE_LOGITS: dict[str, tuple[float, float]] = {
    "CF:q1:N:0": (2.0, 0.0),
    "CF:q2:N:0": (-1.0, 0.0),
    "CF:q3:N:0": (0.5, 0.0),
    "CF:q1:IB:0": (1.0, 0.0),
    "CF:q1:IB:1": (-1.0, 0.0),
    "CF:q2:IB:0": (-0.5, 0.0),
    "CF:q2:IB:1": (0.5, 0.0),
    "CF:q3:IB:0": (0.2, 0.0),
    "CF:q1:CB:0": (2.2, 0.0),
    "CF:q1:CB:1": (1.0, 0.0),
    "CF:q2:CB:0": (2.0, 0.0),
    "CF:q2:CB:1": (-2.0, 0.0),
    "CF:q2:CB:2": (1.0, 0.0),
    "CF:q3:CB:0": (1.05, 0.0),
}

GOLDEN_CF_BASELINE_MARGINS: dict[str, float] = {
    prompt_id: score_a - score_b
    for prompt_id, (score_a, score_b) in GOLDEN_CF_BASELINE_LOGITS.items()
}

# E2E-003: known β=[-0.5,0,0], scales=[1,1,1], selected=[0,1,2], τ=1,
# w_r=w_u=0.5, δ_n=δ_c=0.1, λ_n=λ_c=λ_β=0.1.
# Independent encode/decode algebra on CF:q1:N:0 residual [2,0]:
# z=[1.1,0,0.55], z'=[0.6,0,0.55], Δx=[-0.5,0], logits=[1.5,0].
KNOWN_BETA: tuple[float, float, float] = (-0.5, 0.0, 0.0)
KNOWN_SCALES: tuple[float, float, float] = (1.0, 1.0, 1.0)
KNOWN_SELECTED: tuple[int, int, int] = (0, 1, 2)
GOLDEN_KNOWN_BETA_Q1N_LATENTS = (1.1, 0.0, 0.55)
GOLDEN_KNOWN_BETA_Q1N_LATENTS_PRIME = (0.6, 0.0, 0.55)
GOLDEN_KNOWN_BETA_Q1N_DELTA = (-0.5, 0.0)
GOLDEN_KNOWN_BETA_Q1N_LOGITS = (1.5, 0.0)
# Objective components from the same independent algebra over all CF rows.
GOLDEN_KNOWN_BETA_L_RESIST = 0.793408258204555
GOLDEN_KNOWN_BETA_L_RECOVER = 0.9341394244019439
GOLDEN_KNOWN_BETA_L_BEHAVIOR = 0.8637738413032494
GOLDEN_KNOWN_BETA_L_NEUTRAL = 0.21666666666666667
GOLDEN_KNOWN_BETA_L_CORRECT = 0.4
GOLDEN_KNOWN_BETA_L_BETA = 0.16666666666666666
GOLDEN_KNOWN_BETA_L_TOTAL = 0.9421071746365829
