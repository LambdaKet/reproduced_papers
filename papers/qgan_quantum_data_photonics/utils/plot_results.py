"""Plot the phase-noise, defect-rate and shot-noise results against the paper.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/plot_results.py \\
        papers/qgan_quantum_data_photonics/outdir \\
        --out papers/qgan_quantum_data_photonics/results

Reads the validated sweeps under ``outdir/sweep_*`` (``sweep_status.json`` and
the ``metrics.json`` files it lists) and the TVD run, never logs. Writes each
figure to ``--out`` and to its primary sweep folder. Paper values come from
``paper_values.py`` (read from the paper's figures). Colors: validated
categorical slots (ours blue, ours noise-averaged orange, paper aqua, ceiling
yellow) and a validated one-hue ramp for the noise levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper_values as paper  # noqa: E402

OURS, OURS_NOISY, PAPER, CEILING = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SIGMA_RAMP = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SCOPE = (
    "Simulation, 100 targets per point; training settings calibrated (not given in the paper). "
    "Paper values read from its figures."
)


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _metrics(outdir: Path, sweep: str) -> dict[str, dict]:
    sweep_dir = outdir / f"sweep_{sweep}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep} is {status['status']}, not COMPLETED")
    return {
        run["run_id"]: json.loads((sweep_dir / run["metrics_path"]).read_text())
        for run in status["runs"]
    }


def _save(fig, name: str, out: Path, sweep_dir: Path) -> None:
    for folder in (out, sweep_dir):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            folder / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb"
        )
    plt.close(fig)


def plot_phase_noise(outdir: Path, out: Path) -> None:
    runs = _metrics(outdir, "phase_noise_sweep")
    ceiling = _metrics(outdir, "oracle_ceiling")["ceiling_stuck_0_of_30"]
    sigmas = sorted(paper.PHASE_NOISE_FINAL)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    for color, sigma in zip(SIGMA_RAMP, sigmas, strict=True):
        curve = np.asarray(runs[f"sigma_{sigma}pi"]["mean_fidelity_by_round"])
        left.plot(curve, color=color, linewidth=2, label=f"σ = {sigma}π")
    left.set(xlabel="Round", ylabel="Mean root fidelity", ylim=(0, 1.02), xlim=(0, 400))
    left.legend(
        frameon=False,
        fontsize=8,
        loc="lower right",
        title="Phase noise",
        title_fontsize=8,
    )
    left.set_title(
        "Mean fidelity per round (Fig. 2c)", fontsize=10, color=INK, loc="left"
    )
    _style(left)

    final = [runs[f"sigma_{s}pi"]["final_fidelity"] for s in sigmas]
    right.errorbar(
        sigmas, [f["mean"] for f in final], yerr=[f["std"] for f in final],
        color=OURS, marker="o", markersize=6, linewidth=2, capsize=3,
        label="Ours, intended phases (mean ± std)",
    )  # fmt: skip
    right.plot(
        sigmas, [runs[f"sigma_{s}pi"]["noise_averaged_final_fidelity"]["mean"] for s in sigmas],
        color=OURS_NOISY, marker="s", markersize=6, linewidth=2, label="Ours, noise-averaged",
    )  # fmt: skip
    right.plot(
        sigmas, [paper.PHASE_NOISE_FINAL[s] for s in sigmas], color=PAPER, marker="D",
        markersize=6, linewidth=2, linestyle="--", label="Paper, round 400 (read)",
    )  # fmt: skip
    right.plot(
        sigmas, [ceiling["noise_averaged_best_fidelity"][str(s)]["mean"] for s in sigmas],
        color=CEILING, marker="^", markersize=6, linewidth=2, linestyle=":",
        label="Ceiling under the same noise (ours)",
    )  # fmt: skip
    right.axhline(0.9, color=MUTED, linewidth=1, linestyle="-")
    right.annotate("0.9", (0.0505, 0.9), xytext=(2, 2), textcoords="offset points",
                   fontsize=8, color=MUTED)  # fmt: skip
    right.set(
        xlabel="Phase noise σ (units of π)",
        ylabel="Final root fidelity",
        ylim=(0.7, 1.01),
    )
    right.set_xticks(sigmas)
    right.set_title(
        "Final fidelity against phase noise", fontsize=10, color=INK, loc="left"
    )
    handles, labels = right.get_legend_handles_labels()
    order = [
        labels.index(name)
        for name in sorted(labels, key=lambda n: not n.startswith("Ours, intended"))
    ]
    right.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
                 fontsize=8, loc="lower left")  # fmt: skip
    _style(right)
    fig.suptitle(SCOPE + " 1500 counts per setting.", fontsize=8, color=MUTED, y=0.0)
    _save(fig, "phase_noise", out, outdir / "sweep_phase_noise_sweep")


def plot_defect_rate(outdir: Path, out: Path) -> None:
    runs = {}
    for part in ("low", "mid", "high"):
        runs.update(_metrics(outdir, f"defect_rate_sweep_{part}"))
    ceiling = _metrics(outdir, "oracle_ceiling")
    stuck = paper.DEFECT_STUCK
    mean = np.array([runs[f"stuck_{k}_of_30"]["final_fidelity"]["mean"] for k in stuck])
    std = np.array([runs[f"stuck_{k}_of_30"]["final_fidelity"]["std"] for k in stuck])
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb")
    ax.fill_between(stuck, mean - std, np.minimum(mean + std, 1.0), color=OURS, alpha=0.15,
                    linewidth=0)  # fmt: skip
    ax.plot(stuck, mean, color=OURS, marker="o", markersize=6, linewidth=2,
            label="Ours (mean, band = ± std)")  # fmt: skip
    ax.plot(stuck, paper.DEFECT_FIDELITY, color=PAPER, marker="D", markersize=6, linewidth=2,
            linestyle="--", label="Paper (read)")  # fmt: skip
    ax.plot(
        stuck, [ceiling[f"ceiling_stuck_{k}_of_30"]["best_fidelity"]["mean"] for k in stuck],
        color=CEILING, marker="^", markersize=6, linewidth=2, linestyle=":",
        label="Ceiling: best fidelity the circuit can reach (ours)",
    )  # fmt: skip
    ax.axhline(0.9, color=MUTED, linewidth=1)
    ax.axvline(15, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("half of the 30", (15, 0.32), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=MUTED)  # fmt: skip
    ax.set(xlabel="Stuck generator phase shifters (out of 30)", ylabel="Final root fidelity",
           ylim=(0.3, 1.02))  # fmt: skip
    ax.set_xticks(stuck)
    ax.set_title("Final fidelity against defects (Fig. 3b), σ = 0.02π, 1500 counts",
                 fontsize=10, color=INK, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.suptitle(SCOPE, fontsize=8, color=MUTED, y=0.0)
    _save(fig, "defect_rate", out, outdir / "sweep_defect_rate_sweep_mid")


def plot_shot_noise(outdir: Path, out: Path) -> None:
    runs = _metrics(outdir, "shot_noise_sweep")
    counts = paper.SHOT_COUNTS
    final = [runs[f"counts_{n}"]["final_fidelity"] for n in counts]
    tvd_run = sorted((outdir / "shot_noise_tvd_original").glob("run_*"))[-1]
    tvd = json.loads((tvd_run / "metrics.json").read_text())["mean_tvd_by_count_level"]
    levels = sorted(int(k) for k in tvd)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    left.errorbar(counts, [f["mean"] for f in final], yerr=[f["std"] for f in final],
                  color=OURS, marker="o", markersize=6, linewidth=2, capsize=3,
                  label="Ours (mean ± std)")  # fmt: skip
    left.plot(counts, paper.SHOT_FIDELITY, color=PAPER, marker="D", markersize=6, linewidth=2,
              linestyle="--", label="Paper (read)")  # fmt: skip
    left.set(xlabel="Coincidence counts per setting", ylabel="Final root fidelity",
             ylim=(0.88, 1.005))  # fmt: skip
    left.set_title("Final fidelity against counts (Suppl. Fig. 9b), σ = 0", fontsize=10,
                   color=INK, loc="left")  # fmt: skip
    left.legend(frameon=False, fontsize=8, loc="lower right")
    _style(left)
    right.plot(levels, [tvd[str(n)] for n in levels], color=OURS, marker="o", markersize=6,
               linewidth=2, label="Ours (1000 repeats)")  # fmt: skip
    ends = sorted(paper.TVD_ENDS)
    right.plot(ends, [paper.TVD_ENDS[n] for n in ends], color=PAPER, marker="D", markersize=7,
               linestyle="none", label="Paper, end points (read)")  # fmt: skip
    right.set_xscale("log")
    right.set(xlabel="Total counts (log scale)", ylabel="Mean total variation distance")
    right.set_title("Sampled uniform 16-outcome distribution (Suppl. Fig. 9a)", fontsize=10,
                    color=INK, loc="left")  # fmt: skip
    right.legend(frameon=False, fontsize=8)
    _style(right)
    fig.suptitle(SCOPE, fontsize=8, color=MUTED, y=0.0)
    _save(fig, "shot_noise", out, outdir / "sweep_shot_noise_sweep")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot_phase_noise(args.outdir, args.out)
    plot_defect_rate(args.outdir, args.out)
    plot_shot_noise(args.outdir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
