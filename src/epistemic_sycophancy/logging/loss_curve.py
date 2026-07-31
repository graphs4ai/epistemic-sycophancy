"""Loss-over-trials plot artifact (ORCH-PLOT-001 / DEC-091)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def plot_loss_over_trials(
    trials: Sequence[Mapping[str, Any]],
    output_path: Path | str,
) -> Path | None:
    """Write ``l_total`` and running-best curves vs ``trial_index``.

    Returns the output path, or ``None`` when ``trials`` is empty (no file written).
    """
    if not trials:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [int(row["trial_index"]) for row in trials]
    ys = [float(row["l_total"]) for row in trials]
    running_best: list[float] = []
    best = float("inf")
    for y in ys:
        if y < best:
            best = y
        running_best.append(best)

    kind = trials[0].get("optimizer_kind")
    title = "Loss over trials"
    if kind is not None:
        title = f"Loss over trials ({kind})"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o", label="l_total")
    ax.plot(xs, running_best, marker=".", linestyle="--", label="best_l_total")
    ax.set_xlabel("trial_index")
    ax.set_ylabel("l_total")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
