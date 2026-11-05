"""Plot the worker-scaling result: epoch time saved by 4 workers, and epoch time by execution path.

Usage (from the repository root)::

    python papers/DQuLearn/utils/plot_worker_scaling.py \\
        --sweep papers/DQuLearn/outdir/sweep_worker_scaling_5q

Reads the sweep's ``sweep_summary.json`` (5 qubits). Left: epoch time saved by
the DQuLearn manager with 4 workers instead of 1, as 1 − paired ratio with its
95% Student-t interval over the 3 repetitions, against the paper's Google Cloud
values (Sec. IV-C2 text). Right: epoch time with 4 workers for each execution
path, mean ± std. Saves ``worker_scaling.png`` in the sweep folder and in
``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from build_results import PAPER_CLOUD_5Q, RUNTIME_SCOPE, _comparison, _stat
from plot_style import MUTED, SERIES, apply_style, dots, save, scope_note

LAYERS = (1, 2, 3)
PATHS = (
    ("manager_per_circuit_w4_l{}", "DQuLearn manager, 4 workers"),
    ("pool_w4_l{}", "process pool, 4 processes"),
    ("batched_l{}", "batched, one process"),
    ("manager_cached_w4_l{}", "manager, 4 workers, cached compiler"),
)


def plot(summary: dict):
    apply_style()
    fig, (saved, seconds) = plt.subplots(
        1, 2, figsize=(8.6, 3.7), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    reductions, below, above = [], [], []
    for layers in LAYERS:
        ci = _comparison(summary, f"manager_per_circuit_l{layers}_w4_vs_w1")
        reduction = 100 * (1 - ci["mean"])
        reductions.append(reduction)
        below.append(reduction - 100 * (1 - ci["ci95"][1]))
        above.append(100 * (1 - ci["ci95"][0]) - reduction)
    dots(
        saved,
        [x - 0.1 for x in LAYERS],
        reductions,
        (below, above),
        SERIES[0],
        "this reproduction, one laptop",
    )
    dots(
        saved,
        [x + 0.1 for x in LAYERS],
        [PAPER_CLOUD_5Q["reduction_4_vs_1"][layers] for layers in LAYERS],
        None,
        SERIES[0],
        "paper, Google Cloud",
        hollow=True,
    )
    saved.axhline(75, color=MUTED, linewidth=1, zorder=1)
    saved.text(
        1.05, 76.5, "ideal 4-way split", ha="left", va="bottom", fontsize=8, color=MUTED
    )
    saved.set(
        title="Time saved by 4 workers vs 1 (5 qubits)",
        xlabel="layers",
        ylabel="reduction (%)",
        xlim=(0.55, 3.45),
        ylim=(0, 100),
        xticks=LAYERS,
        yticks=(0, 25, 50, 75, 100),
    )
    saved.legend(loc="lower right")

    offsets = (-0.21, -0.07, 0.07, 0.21)
    for (pattern, label), color, offset in zip(PATHS, SERIES, offsets, strict=True):
        stats = [
            _stat(summary, pattern.format(layers), "epoch_seconds") for layers in LAYERS
        ]
        dots(
            seconds,
            [x + offset for x in LAYERS],
            [s["mean"] for s in stats],
            [s["std"] for s in stats],
            color,
            label,
        )
    seconds.set(
        title="Epoch time by execution path (5 qubits)",
        xlabel="layers",
        ylabel="seconds per epoch",
        xlim=(0.55, 3.45),
        ylim=(0, None),
        xticks=LAYERS,
    )
    seconds.legend(loc="upper left")

    fig.subplots_adjust(bottom=0.27, wspace=0.28)
    scope_note(
        fig,
        f"Scope: {RUNTIME_SCOPE}. Whiskers: left, 95% interval of the paired ratio; right, ±\u00a0std. "
        "Manager and pool rebuild the compiler for every circuit, as QuClassi's code does "
        "through qiskit.execute; the cached compiler is built once per worker; batched compiles "
        "and simulates each bank in one call.",
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweep", type=Path, required=True, help="sweep_worker_scaling_5q folder"
    )
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    args = parser.parse_args()
    summary = json.loads((args.sweep / "sweep_summary.json").read_text())
    if not summary["complete_sweep"]:
        raise SystemExit(f"{args.sweep} is not a complete sweep; refusing to plot it")
    for path in save(plot(summary), "worker_scaling", args.sweep, args.results):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
