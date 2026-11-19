"""Build the result table of the MerLin adaptation from its finished sweeps.

Usage (from the repository root)::

    python papers/DQuLearn/utils/build_photonic_results.py \\
        --sweeps papers/DQuLearn/outdir --results papers/DQuLearn/results

Reads every run's ``metrics.json`` through each sweep's ``sweep_status.json``:
``photonic_accuracy_final`` (three training rules of the photonic model, four
pairs, three seeds), ``photonic_accuracy_parity`` (local, 1 and 2 workers) and,
for comparison, the qubit model of ``accuracy_final`` (Phase 03). Writes
``results/photonic_accuracy.json`` and ``results/photonic_accuracy.md``.

Rules, fixed in the adaptation plan before the runs:

- parity is kept if, on every pair, the mean test accuracies of the
  distributed rule (``dqulearn``) and the non-distributed rule (``quclassi``)
  differ by at most 2 points, and the parity runs give identical parameters;
- other comparisons are paired by seed: the ratio of test accuracies, a
  difference when |mean ratio − 1| > 2 std.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

PAIRS = ("3/9", "3/8", "3/6", "1/5")
RULES = ("dqulearn", "quclassi", "exact_shift")
RULE_NAMES = {
    "dqulearn": "DQuLearn rule, 2 workers",
    "quclassi": "QuClassi rule, one process",
    "exact_shift": "exact photonic shift, 2 workers",
}
PARITY_TOLERANCE_POINTS = 2.0
# DQuLearn Sec. IV-B, two workers.
PAPER_ACCURACY = {"3/9": 97.5, "3/8": 96.2, "3/6": 98.1, "1/5": 98.6}
SCOPE = (
    "partial, simulation only: MerLin adaptation of claim accuracy-parity; 2 photons "
    "in 4 modes (counterpart of QuClassi's 5-qubit circuit), exact probabilities, "
    "QuClassi's preprocessing (4 PCA features), learning rate per rule selected on "
    "the validation split of seed 0, 3 seeds"
)


def _runs(sweep_dir: Path) -> dict:
    """{candidate name: {seed: (labels, metrics)}} for the completed runs of a sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir.name} is {status['status']}, not COMPLETED")
    runs: dict = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            raise ValueError(f"{sweep_dir.name}: {run['run_id']} is {run['status']}")
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        candidate = run["candidate"]
        runs.setdefault(candidate["name"], {})[int(run["seed"])] = (
            candidate["labels"],
            metrics,
        )
    return runs


def _by_rule_and_pair(runs: dict) -> dict:
    """{(rule, pair): {seed: metrics}} for every candidate labelled with a rule."""
    table: dict = {}
    for by_seed in runs.values():
        for seed, (labels, metrics) in by_seed.items():
            if labels.get("rule") is None:
                continue
            table.setdefault((labels["rule"], labels["pair"]), {})[seed] = metrics
    return table


def _test(metrics: dict) -> float:
    return 100.0 * metrics["final"]["test"]["accuracy"]


def _summary(by_seed: dict) -> dict:
    values = np.array([_test(by_seed[seed]) for seed in sorted(by_seed)])
    return {
        "seeds": sorted(by_seed),
        "test_accuracy": values.tolist(),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def _paired_ratio(candidate: dict, reference: dict) -> dict:
    seeds = sorted(set(candidate) & set(reference))
    ratios = np.array([_test(candidate[s]) / _test(reference[s]) for s in seeds])
    mean = float(ratios.mean())
    std = float(ratios.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(ratios) - 1) * std / np.sqrt(len(ratios)))
    return {
        "seeds": seeds,
        "ratios": ratios.tolist(),
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "difference": bool(abs(mean - 1.0) > 2.0 * std),
    }


def _parity_runs(runs: dict) -> dict:
    parameters = {
        name: np.array(next(iter(by_seed.values()))[1]["parameters"])
        for name, by_seed in runs.items()
    }
    reference = parameters["local"]
    return {
        "test_accuracy": {
            name: _test(next(iter(by_seed.values()))[1])
            for name, by_seed in runs.items()
        },
        "max_abs_parameter_difference_vs_local": {
            name: float(np.max(np.abs(values - reference)))
            for name, values in parameters.items()
        },
    }


def build(sweeps_dir: Path) -> tuple[dict, str]:
    photonic = _by_rule_and_pair(_runs(sweeps_dir / "sweep_photonic_accuracy_final"))
    qubit = _by_rule_and_pair(_runs(sweeps_dir / "sweep_accuracy_final"))
    parity = _parity_runs(_runs(sweeps_dir / "sweep_photonic_accuracy_parity"))

    rows = {}
    for pair in PAIRS:
        rule_stats = {rule: _summary(photonic[(rule, pair)]) for rule in RULES}
        gap = rule_stats["dqulearn"]["mean"] - rule_stats["quclassi"]["mean"]
        rows[pair] = {
            "photonic": rule_stats,
            "qubit_dqulearn": _summary(qubit[("dqulearn", pair)]),
            "paper": PAPER_ACCURACY[pair],
            "dqulearn_minus_quclassi_points": gap,
            "photonic_vs_qubit_dqulearn": _paired_ratio(
                photonic[("dqulearn", pair)], qubit[("dqulearn", pair)]
            ),
            "exact_shift_vs_dqulearn": _paired_ratio(
                photonic[("exact_shift", pair)], photonic[("dqulearn", pair)]
            ),
        }
    any_metrics = next(iter(photonic[("dqulearn", "3/6")].values()))
    circuits = {
        rule: next(iter(photonic[(rule, "3/6")].values()))["circuits_per_sample"]
        for rule in RULES
    }
    lrs = {rule: next(iter(photonic[(rule, "3/6")].values()))["lr"] for rule in RULES}
    wall_clock = {
        rule: float(
            np.mean(
                [
                    metrics["seconds_total"]
                    for pair in PAIRS
                    for metrics in photonic[(rule, pair)].values()
                ]
            )
        )
        for rule in RULES
    }
    seeds = sorted({seed for by_seed in photonic.values() for seed in by_seed})
    parity_kept = all(
        abs(row["dqulearn_minus_quclassi_points"]) <= PARITY_TOLERANCE_POINTS
        for row in rows.values()
    ) and all(
        value == 0.0
        for value in parity["max_abs_parameter_difference_vs_local"].values()
    )
    result = {
        "scope": SCOPE,
        "hardware": any_metrics["photonic"],
        "learning_rates": lrs,
        "circuits_per_sample": circuits,
        "mean_run_seconds": wall_clock,
        "seeds": seeds,
        "parity_tolerance_points": PARITY_TOLERANCE_POINTS,
        "parity_kept": parity_kept,
        "parity_runs": parity,
        "pairs": rows,
    }
    return result, _markdown(result)


def _pm(summary: dict) -> str:
    return f"{summary['mean']:.1f} ± {summary['std']:.1f}"


def _ratio(ratio: dict) -> str:
    low, high = ratio["ci95"]
    tag = "difference" if ratio["difference"] else "no difference"
    return f"{ratio['mean']:.3f} [{low:.3f}, {high:.3f}] ({tag})"


def _markdown(result: dict) -> str:
    hw = result["hardware"]
    lines = [
        "# MerLin adaptation: accuracy parity with a photonic workload",
        "",
        f"Scope: {result['scope']}.",
        "",
        "Test accuracy (%), mean ± std over 3 seeds. Photonic model: "
        f"{hw['photons']} photons, {hw['modes']} modes, input {hw['input_state']}, "
        f"{hw['encoding']}; {hw['trainable']}; {hw['measurement']}; detector "
        f"{hw['detector_model']}; postselection {hw['postselection']}; "
        f"{hw['simulator']} (merlinquantum {hw['merlinquantum']}, perceval-quandela "
        f"{hw['perceval_quandela']}). Learning rates: "
        + ", ".join(
            f"{RULE_NAMES[r]} {lr:g}" for r, lr in result["learning_rates"].items()
        )
        + ".",
        "",
        "| Pair | "
        + " | ".join(f"Photonic, {RULE_NAMES[r]}" for r in RULES)
        + " | Qubit model, DQuLearn rule (Phase 03) | Paper |",
        "|---|---|---|---|---|---|",
    ]
    for pair, row in result["pairs"].items():
        cells = [_pm(row["photonic"][r]) for r in RULES]
        lines.append(
            f"| {pair} | "
            + " | ".join(cells)
            + f" | {_pm(row['qubit_dqulearn'])} | {row['paper']} |"
        )
    lines += [
        "",
        "Parity (rule fixed before the runs: |DQuLearn rule − QuClassi rule| ≤ "
        f"{result['parity_tolerance_points']:g} points on every pair, and identical "
        "parameters for local, 1-worker and 2-worker training): "
        + ("kept." if result["parity_kept"] else "not kept."),
        "",
        "| Pair | DQuLearn − QuClassi rule (points) | Photonic / qubit, DQuLearn rule "
        "(paired ratio [95% interval]) | Exact shift / two-point (paired ratio) |",
        "|---|---|---|---|",
    ]
    for pair, row in result["pairs"].items():
        lines.append(
            f"| {pair} | {row['dqulearn_minus_quclassi_points']:+.2f} | "
            f"{_ratio(row['photonic_vs_qubit_dqulearn'])} | "
            f"{_ratio(row['exact_shift_vs_dqulearn'])} |"
        )
    parity = result["parity_runs"]
    lines += [
        "",
        "Parity runs (3/6, seed 0, DQuLearn rule): largest parameter difference "
        "against local training: "
        + ", ".join(
            f"{name} {value:g}"
            for name, value in parity["max_abs_parameter_difference_vs_local"].items()
        )
        + "; test accuracy "
        + ", ".join(
            f"{name} {value:.2f}%" for name, value in parity["test_accuracy"].items()
        )
        + ".",
        "",
        "Circuits per sample (each one a task for a worker): "
        + ", ".join(
            f"{RULE_NAMES[r]} {n}" for r, n in result["circuits_per_sample"].items()
        )
        + ". A difference is |mean ratio − 1| > 2 std of the paired ratios.",
        "",
        "Gradients: parameter-shift estimates from exact probabilities; the two-point "
        "±π/2 estimate is not the derivative for 2 photons, the exact shift rule is. "
        "Seeds: "
        + ", ".join(str(seed) for seed in result["seeds"])
        + ". Mean wall-clock "
        "per run (25 epochs and evaluations, this laptop's CPU): "
        + ", ".join(
            f"{RULE_NAMES[r]} {seconds:.0f} s"
            for r, seconds in result["mean_run_seconds"].items()
        )
        + ". Simulated, not hardware-executed; no loss or noise.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result, markdown = build(args.sweeps)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "photonic_accuracy.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.results / "photonic_accuracy.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {args.results / 'photonic_accuracy.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
