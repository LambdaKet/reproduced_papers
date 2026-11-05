"""Plot the accuracy result: test accuracy per MNIST digit pair, against the paper and QuClassi.

Usage (from the repository root)::

    python papers/DQuLearn/utils/plot_accuracy.py \\
        --sweep papers/DQuLearn/outdir/sweep_accuracy_final

Reads the sweep's ``sweep_summary.json``: test accuracy of the DQuLearn and
QuClassi training rules and of logistic regression on the same features, mean ±
std over seeds 0, 1, 2. Paper values are DQuLearn's Sec. IV-B (2 workers) and
QuClassi's single-layer model (its Fig. 9, measured on the rendered page).
Saves ``accuracy.png`` in the sweep folder and in ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from build_results import ACCURACY_SCOPE, PAPER_ACCURACY, _stat
from plot_style import SERIES, apply_style, dots, save, scope_note

PAIRS = ("3/9", "3/8", "3/6", "1/5")
MODELS = (
    ("dqulearn", "DQuLearn training, 2 workers"),
    ("quclassi", "QuClassi training, one process"),
    ("logistic", "logistic regression"),
)
PAPER = ("paper: DQuLearn", "paper: QuClassi (its Fig. 9)")


def plot(summary: dict):
    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    positions = range(len(PAIRS))
    offsets = (-0.18, 0.0, 0.18)

    for (model, label), color, offset in zip(MODELS, SERIES[:3], offsets, strict=True):
        stats = [
            _stat(summary, f"{model}_pair{pair.replace('/', '')}", "test_accuracy")
            for pair in PAIRS
        ]
        dots(
            ax,
            [x + offset for x in positions],
            [100 * s["mean"] for s in stats],
            [100 * s["std"] for s in stats],
            color,
            label,
        )
    for index, (label, color, offset) in enumerate(
        zip(PAPER, SERIES[:2], offsets[:2], strict=True)
    ):
        dots(
            ax,
            [x + offset for x in positions],
            [PAPER_ACCURACY[pair][index] for pair in PAIRS],
            None,
            color,
            label,
            hollow=True,
        )
    ax.set(
        title="Test accuracy per MNIST digit pair (5 qubits, 1 layer)",
        xlabel="digit pair",
        ylabel="test accuracy (%)",
        xticks=positions,
        xticklabels=PAIRS,
        xlim=(-0.5, len(PAIRS) - 0.5),
        ylim=(65, 100),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.subplots_adjust(right=0.72, bottom=0.24)
    scope_note(
        fig,
        f"Scope: {ACCURACY_SCOPE}. Whiskers: ±\u00a0std. QuClassi values measured on the rendered "
        "Fig. 9 of the QuClassi paper (single-layer model).",
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweep", type=Path, required=True, help="sweep_accuracy_final folder"
    )
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    args = parser.parse_args()
    summary = json.loads((args.sweep / "sweep_summary.json").read_text())
    if not summary["complete_sweep"]:
        raise SystemExit(f"{args.sweep} is not a complete sweep; refusing to plot it")
    for path in save(plot(summary), "accuracy", args.sweep, args.results):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
