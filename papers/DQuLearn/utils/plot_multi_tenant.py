"""Plot the multi-tenant result: completion time of each job, and the multi- vs single-tenant change.

Usage (from the repository root)::

    python papers/DQuLearn/utils/plot_multi_tenant.py \\
        --sweep papers/DQuLearn/outdir/sweep_multi_tenant

Reads the sweep's ``sweep_summary.json``. Left: completion time of each of the
four concurrent jobs from their common start, per tenancy mode, mean ± std over
3 seeds. Right: change of each job's completion time, multi- vs single-tenant,
as paired ratio − 1 with its 95% Student-t interval, against the paper's text
(Sec. IV-C2) and its Fig. 6 bars (measured on the rendered page). Saves
``multi_tenant.png`` in the sweep folder and in ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from build_results import PAPER_MULTI_TENANT, RUNTIME_SCOPE, _comparison, _stat
from plot_style import BASELINE, SERIES, apply_style, dots, save, scope_note

JOBS = ("5Q/1L", "5Q/2L", "7Q/1L", "7Q/2L")
MODES = (
    ("multi_least_cpu", "multi-tenant, lowest CPU (DQuLearn)"),
    ("single_least_cpu", "single-tenant, one circuit per worker"),
    ("sequential_least_cpu", "jobs one after another"),
    ("multi_round_robin", "multi-tenant, round-robin"),
)


def plot(summary: dict):
    apply_style()
    fig, (completion, change) = plt.subplots(1, 2, figsize=(8.6, 3.9))
    positions = range(len(JOBS))

    offsets = (-0.21, -0.07, 0.07, 0.21)
    for (mode, label), color, offset in zip(MODES, SERIES, offsets, strict=True):
        stats = [
            _stat(summary, mode, f"completion_{job.replace('/', '_')}") for job in JOBS
        ]
        dots(
            completion,
            [x + offset for x in positions],
            [s["mean"] for s in stats],
            [s["std"] for s in stats],
            color,
            label,
        )
    completion.set(
        title="Completion time of each job",
        xlabel="job (qubits / layers)",
        ylabel="seconds from the common start",
        xticks=positions,
        xticklabels=JOBS,
        xlim=(-0.5, len(JOBS) - 0.5),
        ylim=(0, 220),
    )
    completion.legend(loc="upper left")

    ours, below, above = [], [], []
    for job in JOBS:
        ci = _comparison(summary, f"multi_vs_single_completion_{job.replace('/', '_')}")
        ours.append(100 * (ci["mean"] - 1))
        below.append(100 * (ci["mean"] - ci["ci95"][0]))
        above.append(100 * (ci["ci95"][1] - ci["mean"]))
    dots(
        change,
        [x - 0.14 for x in positions],
        ours,
        (below, above),
        SERIES[0],
        "this reproduction",
    )
    text = [
        (x, PAPER_MULTI_TENANT[job]["text_reduction"])
        for x, job in zip(positions, JOBS, strict=True)
    ]
    dots(
        change,
        [x for x, value in text if value is not None],
        [-value for _, value in text if value is not None],
        None,
        SERIES[0],
        "paper, text",
        hollow=True,
    )
    bars = [PAPER_MULTI_TENANT[job]["bars"] for job in JOBS]
    dots(
        change,
        [x + 0.14 for x in positions],
        [100 * (multi / single - 1) for single, multi in bars],
        None,
        SERIES[0],
        "paper, Fig. 6 bars",
        hollow=True,
        marker="D",
    )
    change.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
    change.set(
        title="Change in completion time, multi- vs single-tenant",
        xlabel="job (qubits / layers)",
        ylabel="change (%), below 0 = earlier",
        xticks=positions,
        xticklabels=JOBS,
        xlim=(-0.5, len(JOBS) - 0.5),
        ylim=(-100, 60),
    )
    change.legend(loc="lower right")

    fig.subplots_adjust(bottom=0.25, wspace=0.28)
    scope_note(
        fig,
        f"Scope: {RUNTIME_SCOPE}. Four jobs submitted together to workers of 5, 10, 15 and 20 "
        "qubits. Whiskers: left, ±\u00a0std over 3 seeds (the seed shuffles the submission order); "
        "right, 95% interval of the paired ratio. Paper bars measured on the rendered Fig. 6.",
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweep", type=Path, required=True, help="sweep_multi_tenant folder"
    )
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    args = parser.parse_args()
    summary = json.loads((args.sweep / "sweep_summary.json").read_text())
    if not summary["complete_sweep"]:
        raise SystemExit(f"{args.sweep} is not a complete sweep; refusing to plot it")
    for path in save(plot(summary), "multi_tenant", args.sweep, args.results):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
