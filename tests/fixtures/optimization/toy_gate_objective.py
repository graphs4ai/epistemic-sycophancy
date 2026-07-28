"""Pinned affine-margin toy for Phase H optimizer gate (DEC-036).

Hand-built so suppressing β improves the hinge objective from β=0.
Never regenerate expected improvement from production outputs as a golden;
the gate only requires L_final < L_initial within bounds.
"""

from __future__ import annotations

import torch

# Three questions; Q+={q1,q3}, Q-={q2}
GATE_Q_PLUS = frozenset({"q1", "q3"})
GATE_Q_MINUS = frozenset({"q2"})
GATE_QUESTION_IDS = ("q1", "q2", "q3")

# Affine margins: M = const + jac · β  (β length-3)
# Jac columns chosen so negative β increases IB/CB truthful margins on eligible sets.
GATE_IB_MARGIN_CONST = {
    "q1": [0.0, -0.5],
    "q3": [-0.2],
}
GATE_IB_MARGIN_JAC = {
    "q1": [
        torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([-0.5, 0.0, 0.0], dtype=torch.float64),
    ],
    "q3": [torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)],
}
GATE_CB_MARGIN_CONST = {
    "q1": [1.0],
    "q2": [-0.5, -1.0],
    "q3": [0.5],
}
GATE_CB_MARGIN_JAC = {
    "q1": [torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float64)],
    "q2": [
        torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, -0.5], dtype=torch.float64),
    ],
    "q3": [torch.tensor([0.0, -0.3, 0.0], dtype=torch.float64)],
}
GATE_BASELINE_CB = {"q1": [1.2], "q3": [0.6]}
GATE_BASELINE_NEUTRAL = {"q1": 1.0, "q2": -0.5, "q3": 0.4}
GATE_NEUTRAL_CONST = {"q1": 1.0, "q2": -0.5, "q3": 0.4}
GATE_NEUTRAL_JAC = {
    "q1": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
    "q2": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
    "q3": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
}

GATE_TAU = 1.0
GATE_W_R = 0.5
GATE_W_U = 0.5
GATE_DELTA_N = 0.1
GATE_DELTA_C = 0.1
GATE_LAMBDA_N = 1.0
GATE_LAMBDA_C = 1.0
GATE_LAMBDA_BETA = 0.01
GATE_BETA0 = [0.0, 0.0, 0.0]
GATE_BETA_LOWER = -2.0
GATE_BETA_UPPER = 0.0
GATE_N_STEPS = 25
GATE_ADAM_LR = 0.1
