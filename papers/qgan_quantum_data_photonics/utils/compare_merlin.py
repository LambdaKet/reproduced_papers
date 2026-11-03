"""Compare the MerLin adaptation (phase 04) with the numpy reproduction (phase 03).

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/compare_merlin.py \\
        papers/qgan_quantum_data_photonics/outdir \\
        --out papers/qgan_quantum_data_photonics/results

Reads the ``metrics.json`` and ``trajectories.npz`` files listed by the
``sweep_status.json`` of the phase-03 sweeps and of the three MerLin sweeps,
never logs. Applies the tolerances set before the MerLin runs:

- equivalence: >= 90 % of targets with a final fidelity within 1e-6 of
  phase 03 and condition means within 0.005;
- MerLin-native noise: condition means within 0.02 of phase 03 and the same
  predeclared claim outcomes (>= 0.90 for sigma <= 0.03 pi; >= 0.90 up to
  14/30 stuck).

Paired differences use the same targets in both runs (the native-noise runs do
not share the noise draws); 95 % percentile bootstrap over targets, 10,000
resamples, generator seed 0. Writes ``merlin_comparison.json`` and ``.md`` and
the figure ``merlin_comparison.png`` to ``--out``.
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

NUMPY_COLOR, MERLIN_COLOR, PAPER_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"
SIGMAS = (0.01, 0.02, 0.03, 0.04, 0.05)
NATIVE_STUCK = (0, 8, 14, 16, 20)
NATIVE_TOLERANCE = 0.02
LARGE_DIFFERENCE = 0.1
EQUIVALENCE_LABELS = {
    "merlin_injected_sigma_0.01pi": "σ = 0.01π",
    "merlin_injected_sigma_0.05pi": "σ = 0.05π",
    "merlin_injected_stuck_16_of_30": "16 / 30 stuck, σ = 0.02π",
}


def _runs(outdir: Path, sweep: str) -> dict[str, tuple[dict, Path]]:
    sweep_dir = outdir / f"sweep_{sweep}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep} is {status['status']}, not COMPLETED")
    return {
        run["run_id"]: (
            json.loads((sweep_dir / run["metrics_path"]).read_text()),
            sweep_dir / run["run_dir"],
        )
        for run in status["runs"]
    }


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, diff.size, (10_000, diff.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    large = np.abs(diff) > LARGE_DIFFERENCE
    return {
        "mean": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "median": float(np.median(diff)),
        "n": int(diff.size),
        # How much of the mean comes from targets whose runs end far apart.
        "targets_differing_by_more_than_0.1": int(large.sum()),
        "their_share_of_the_mean": float(diff[large].sum() / diff.size),
    }


def _final(metrics: dict) -> np.ndarray:
    return np.asarray(metrics["final_fidelity_per_seed"])


def compare(outdir: Path) -> dict:
    noise = _runs(outdir, "phase_noise_sweep")
    defects = {}
    for part in ("low", "mid", "high"):
        defects.update(_runs(outdir, f"defect_rate_sweep_{part}"))
    equivalence = _runs(outdir, "merlin_equivalence")
    native_noise = _runs(outdir, "merlin_native_phase_noise")
    native_defects = _runs(outdir, "merlin_native_defects")

    pairs = {
        "merlin_injected_sigma_0.01pi": noise["sigma_0.01pi"],
        "merlin_injected_sigma_0.05pi": noise["sigma_0.05pi"],
        "merlin_injected_stuck_16_of_30": defects["stuck_16_of_30"],
    }
    equivalence_rows = []
    for name, (ref_metrics, ref_dir) in pairs.items():
        merlin_metrics, merlin_dir = equivalence[name]
        if ref_metrics["seeds"] != merlin_metrics["seeds"]:
            raise ValueError(f"{name}: targets differ from phase 03")
        a, b = _final(ref_metrics), _final(merlin_metrics)
        ta = np.load(ref_dir / "trajectories.npz")["fidelity"]
        tb = np.load(merlin_dir / "trajectories.npz")["fidelity"]
        share = float(np.mean(np.abs(a - b) <= 1e-6))
        equivalence_rows.append(
            {
                "condition": name,
                "targets": int(a.size),
                "share_within_1e-6": share,
                "mean_difference": float(b.mean() - a.mean()),
                "max_trajectory_difference": float(np.abs(ta - tb).max()),
                "holds": bool(share >= 0.9 and abs(b.mean() - a.mean()) <= 0.005),
            }
        )

    def native_row(label, ref_metrics, merlin_metrics):
        a, b = _final(ref_metrics), _final(merlin_metrics)
        if ref_metrics["seeds"] != merlin_metrics["seeds"]:
            raise ValueError(f"{label}: targets differ from phase 03")
        return {
            "condition": label,
            "numpy_mean": float(a.mean()),
            "numpy_std": float(a.std(ddof=1)),
            "merlin_mean": float(b.mean()),
            "merlin_std": float(b.std(ddof=1)),
            "merlin_minus_numpy": _paired(b, a),
            "within_tolerance": bool(abs(b.mean() - a.mean()) <= NATIVE_TOLERANCE),
            "merlin_seconds": merlin_metrics["seconds"],
            "numpy_seconds": ref_metrics["seconds"],
        }

    noise_rows = [
        native_row(
            f"sigma_{s}pi",
            noise[f"sigma_{s}pi"][0],
            native_noise[f"merlin_phase_error_sigma_{s}pi"][0],
        )
        | {"sigma_pi": s, "paper": paper.PHASE_NOISE_FINAL[s]}
        for s in SIGMAS
    ]
    defect_rows = [
        native_row(
            f"stuck_{k}_of_30",
            defects[f"stuck_{k}_of_30"][0],
            native_defects[f"merlin_phase_error_stuck_{k}_of_30"][0],
        )
        | {
            "stuck_of_30": k,
            "paper": paper.DEFECT_FIDELITY[paper.DEFECT_STUCK.index(k)],
        }
        for k in NATIVE_STUCK
    ]

    def claim_rule(rows, key, limit):
        covered = [r for r in rows if r[key] <= limit]
        numpy_holds = all(r["numpy_mean"] >= 0.9 for r in covered)
        merlin_holds = all(r["merlin_mean"] >= 0.9 for r in covered)
        return {
            "numpy_holds": numpy_holds,
            "merlin_holds": merlin_holds,
            "same_outcome": numpy_holds == merlin_holds,
        }

    hardware = next(iter(native_noise.values()))[0]["hardware"]
    return {
        "equivalence": {
            "rows": equivalence_rows,
            "holds": all(r["holds"] for r in equivalence_rows),
        },
        "native_phase_noise": {
            "rows": noise_rows,
            "claim_rule": {"statement": "mean ≥ 0.90 for σ ≤ 0.03π"}
            | claim_rule(noise_rows, "sigma_pi", 0.03),
            "conditions_within_tolerance": sum(
                r["within_tolerance"] for r in noise_rows
            ),
        },
        "native_defects": {
            "rows": defect_rows,
            "claim_rule": {"statement": "mean ≥ 0.90 up to 14/30 stuck"}
            | claim_rule(defect_rows, "stuck_of_30", 14),
            "conditions_within_tolerance": sum(
                r["within_tolerance"] for r in defect_rows
            ),
        },
        "tolerance": NATIVE_TOLERANCE,
        "hardware": hardware,
    }


def markdown(result: dict) -> str:
    lines = [
        "### Equivalence: MerLin chip with the phase-03 noise draws (100 targets each)",
        "",
        "Final root fidelity per target, MerLin against numpy.",
        "",
        "| Condition | Targets identical (within 1e-6) | Mean difference "
        "| Largest per-round difference | Holds |",
        "|---|---:|---:|---:|---|",
    ]
    for r in result["equivalence"]["rows"]:
        lines.append(
            f"| {EQUIVALENCE_LABELS[r['condition']]} | {r['share_within_1e-6']:.0%} "
            f"| {r['mean_difference']:.1e} | {r['max_trajectory_difference']:.1e} "
            f"| {'yes' if r['holds'] else 'no'} |"
        )
    for key, title, header, label in (
        (
            "native_phase_noise",
            "MerLin phase noise, Fig. 2c conditions (1500 counts)",
            "Phase noise σ",
            lambda r: f"{r['sigma_pi']}π",
        ),
        (
            "native_defects",
            "MerLin phase noise, Fig. 3b conditions (σ = 0.02π, 1500 counts)",
            "Stuck phase shifters",
            lambda r: f"{r['stuck_of_30']} / 30",
        ),
    ):
        block = result[key]
        lines += [
            "",
            f"### {title}",
            "",
            "Final root fidelity, mean ± std over the same 100 targets.",
            "",
            f"| {header} | Numpy chip | MerLin chip | MerLin − numpy [95 % CI] | Paper (read) |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in block["rows"]:
            d = r["merlin_minus_numpy"]
            lines.append(
                f"| {label(r)} | {r['numpy_mean']:.3f} ± {r['numpy_std']:.3f} "
                f"| {r['merlin_mean']:.3f} ± {r['merlin_std']:.3f} "
                f"| {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                f"| {r['paper']:.3f} |"
            )
        rule = block["claim_rule"]
        lines += [
            "",
            f"Claim rule ({rule['statement']}): numpy {'holds' if rule['numpy_holds'] else 'fails'}, "
            f"MerLin {'holds' if rule['merlin_holds'] else 'fails'}. "
            f"Means within {result['tolerance']}: {block['conditions_within_tolerance']} "
            f"of {len(block['rows'])} conditions.",
        ]
    return "\n".join(lines)


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot(result: dict, out: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7), sharex="col", facecolor="#fcfcfb",
        gridspec_kw={"height_ratios": [2, 1]},
    )  # fmt: skip
    panels = (
        ("native_phase_noise", "sigma_pi", "Phase noise σ (units of π)",
         "Fig. 2c conditions, 1500 counts", (0.7, 1.01)),
        ("native_defects", "stuck_of_30", "Stuck generator phase shifters (out of 30)",
         "Fig. 3b conditions, σ = 0.02π, 1500 counts", (0.3, 1.02)),
    )  # fmt: skip
    tolerance = result["tolerance"]
    for column, (key, label, xlabel, title, ylim) in enumerate(panels):
        rows = result[key]["rows"]
        x = [r[label] for r in rows]
        top, bottom = axes[0, column], axes[1, column]
        top.plot(x, [r["numpy_mean"] for r in rows], color=NUMPY_COLOR, marker="o",
                 markersize=6, linewidth=2, label="Reproduction, numpy chip")  # fmt: skip
        top.plot(x, [r["merlin_mean"] for r in rows], color=MERLIN_COLOR, marker="s",
                 markersize=6, linewidth=2, linestyle="--",
                 label="MerLin chip, MerLin phase noise")  # fmt: skip
        top.plot(x, [r["paper"] for r in rows], color=PAPER_COLOR, marker="D", markersize=6,
                 linewidth=2, linestyle=":", label="Paper (read)")  # fmt: skip
        top.axhline(0.9, color=MUTED, linewidth=1)
        top.annotate("claim threshold 0.9", (x[0], 0.9), xytext=(2, 3),
                     textcoords="offset points", fontsize=8, color=MUTED)  # fmt: skip
        top.set(ylabel="Final root fidelity (mean)", ylim=ylim)
        top.set_title(title, fontsize=10, color=INK, loc="left")
        top.legend(frameon=False, fontsize=8, loc="lower left")
        _style(top)

        diff = [r["merlin_minus_numpy"] for r in rows]
        means = np.array([d["mean"] for d in diff])
        errors = np.abs(np.array([d["ci95"] for d in diff]).T - means)
        span = (min(x) - 0.05 * (max(x) - min(x)), max(x) + 0.05 * (max(x) - min(x)))
        bottom.fill_between(span, -tolerance, tolerance, color=BAND, linewidth=0,
                            label=f"Tolerance ±{tolerance} (set before the runs)")  # fmt: skip
        bottom.axhline(0, color=MUTED, linewidth=1)
        bottom.errorbar(x, means, yerr=errors, color=MERLIN_COLOR, marker="s", markersize=6,
                        linewidth=0, elinewidth=2, capsize=3,
                        label="Mean over the same 100 targets (95 % CI)")  # fmt: skip
        bottom.set(
            xlabel=xlabel, ylabel="MerLin − numpy", ylim=(-0.07, 0.07), xlim=span
        )
        bottom.set_xticks(x)
        _style(bottom)
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "Simulation, 100 targets per point. The MerLin runs draw their own phase noise "
        "(uniform, same standard deviation), so single trajectories differ from numpy. "
        "Paper values read from its figures.",
        fontsize=8, color=MUTED, y=0.0,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(
        out / "merlin_comparison.png", dpi=160, bbox_inches="tight", facecolor="#fcfcfb"
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.outdir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "merlin_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    text = markdown(result)
    (args.out / "merlin_comparison.md").write_text(text + "\n")
    plot(result, args.out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
