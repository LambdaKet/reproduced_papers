"""Build the Phase 5 comparison tables from the finished sweeps.

Usage (from the repository root)::

    python papers/DQuLearn/utils/build_comparison.py \\
        --sweeps papers/DQuLearn/outdir --results papers/DQuLearn/results

For each claim axis (accuracy kept, epoch time with more workers, completion
time under multi-tenancy) it reads every run's ``metrics.json`` through the
sweep's ``sweep_status.json``, checks that the completed runs equal the sweep's
expected runs and that every value used is finite, and writes
``results/comparison.json`` and ``results/comparison.md``.

Rules (frozen in the Phase 5 comparison plan): mean and sample std per cell;
comparisons paired by seed or repetition as ratios, with a 95% Student-t
interval; two systems differ when |mean ratio − 1| > 2 std; parity when the
DQuLearn and non-distributed accuracies differ by at most 2 points per pair.
Paper values are listed beside local values, never ranked with them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

PAIRS = ("3/9", "3/8", "3/6", "1/5")
JOBS = ("5Q/1L", "5Q/2L", "7Q/1L", "7Q/2L")
PARITY_TOLERANCE_POINTS = 2.0
EXPECTED_RUNS = {
    "accuracy_final": 36,
    "accuracy_parity": 3,
    "photonic_accuracy_final": 36,
    "photonic_accuracy_parity": 3,
    "worker_scaling_5q": 90,
    "worker_scaling_7q": 36,
    "multi_tenant": 12,
    "multi_tenant_variants_rerun": 15,
}
# DQuLearn Sec. IV-B (2 workers), Sec. IV-C2 (4 vs 1 worker, Google Cloud, text)
# and Fig. 6 (text values; bars in results/multi_tenant.md).
PAPER = {
    "accuracy": {"3/9": 97.5, "3/8": 96.2, "3/6": 98.1, "1/5": 98.6},
    "saving_4_vs_1_workers": {1: 27.1, 2: 37.3, 3: 43.2},
    "multi_tenant_change": {"5Q/1L": -68.7, "7Q/2L": -8.2},
}

ACCURACY_SYSTEMS = (
    # (key, sweep, rule label, role, provenance, parameters per model)
    (
        "qubit_dqulearn",
        "accuracy_final",
        "dqulearn",
        "reproduced method",
        "local",
        "4 per class",
    ),
    (
        "qubit_quclassi",
        "accuracy_final",
        "quclassi",
        "paper's baseline (non-distributed)",
        "local",
        "4 per class",
    ),
    (
        "logistic",
        "accuracy_final",
        "logistic_regression",
        "added classical baseline",
        "local",
        "5",
    ),
    (
        "photonic_dqulearn",
        "photonic_accuracy_final",
        "dqulearn",
        "MerLin adaptation",
        "local",
        "12 per class",
    ),
    (
        "photonic_quclassi",
        "photonic_accuracy_final",
        "quclassi",
        "MerLin adaptation, non-distributed",
        "local",
        "12 per class",
    ),
    (
        "photonic_exact_shift",
        "photonic_accuracy_final",
        "exact_shift",
        "MerLin adaptation, exact shift",
        "local",
        "12 per class",
    ),
)
SYSTEM_NAMES = {
    "qubit_dqulearn": "DQuLearn, qubit model, 2 workers",
    "qubit_quclassi": "QuClassi rule, qubit model, one process",
    "logistic": "Logistic regression, same 4 features",
    "photonic_dqulearn": "DQuLearn, photonic model, 2 workers",
    "photonic_quclassi": "QuClassi rule, photonic model, one process",
    "photonic_exact_shift": "Exact photonic shift, 2 workers",
}


def _finite(value: float, where: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in {where}")
    return value


def _load(sweeps_dir: Path, sweep: str) -> list[tuple[dict, int, dict]]:
    """(candidate, seed, metrics) for every run; fails unless all expected runs completed."""
    folder = sweeps_dir / f"sweep_{sweep}"
    status = json.loads((folder / "sweep_status.json").read_text())
    runs = status["runs"]
    completed = [run for run in runs if run["status"] == "COMPLETED"]
    keys = [(run["candidate"]["name"], run["seed"]) for run in completed]
    if status["status"] != "COMPLETED" or len(completed) != EXPECTED_RUNS[sweep]:
        raise ValueError(
            f"{sweep}: {status['status']}, {len(completed)} completed runs, "
            f"{EXPECTED_RUNS[sweep]} expected"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"{sweep}: duplicate (candidate, seed) runs")
    return [
        (
            run["candidate"],
            int(run["seed"]),
            json.loads((folder / run["metrics_path"]).read_text()),
        )
        for run in completed
    ]


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
    }


def _ratio(candidate: dict, reference: dict) -> dict:
    """Paired ratio candidate / reference over common seeds (dicts seed -> value)."""
    seeds = sorted(set(candidate) & set(reference))
    ratios = np.array([candidate[s] / reference[s] for s in seeds])
    mean, std = float(ratios.mean()), float(ratios.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(ratios) - 1) * std / np.sqrt(len(ratios)))
    return {
        "n": len(seeds),
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "difference": bool(abs(mean - 1.0) > 2.0 * std),
    }


def accuracy(sweeps_dir: Path) -> dict:
    """Test accuracy (%) per system and pair, seconds per run, and paired comparisons."""
    by_system: dict = {
        key: {pair: {} for pair in PAIRS} for key, *_ in ACCURACY_SYSTEMS
    }
    seconds: dict = {key: [] for key, *_ in ACCURACY_SYSTEMS}
    loaded = {
        sweep: _load(sweeps_dir, sweep) for sweep in {s[1] for s in ACCURACY_SYSTEMS}
    }
    for key, sweep, rule, *_ in ACCURACY_SYSTEMS:
        for candidate, seed, metrics in loaded[sweep]:
            labels = candidate["labels"]
            if labels.get("rule") != rule:
                continue
            test = (
                metrics["final"]["test"]["accuracy"]
                if "final" in metrics
                else metrics["test_accuracy"]
            )
            by_system[key][labels["pair"]][seed] = 100.0 * _finite(
                test, candidate["name"]
            )
            seconds[key].append(_finite(metrics["seconds_total"], candidate["name"]))
    rows = {}
    for key, sweep, _rule, role, provenance, parameters in ACCURACY_SYSTEMS:
        runs = sum(len(v) for v in by_system[key].values())
        rows[key] = {
            "name": SYSTEM_NAMES[key],
            "role": role,
            "provenance": provenance,
            "sweep": sweep,
            "runs": runs,
            "expected_runs": 12,
            "parameters": parameters,
            "pairs": {
                pair: _stats(list(by_system[key][pair].values())) for pair in PAIRS
            },
            "mean_run_seconds": float(np.mean(seconds[key])),
        }
    comparisons = {}
    for name, cand, ref in (
        ("DQuLearn vs QuClassi rule, qubit", "qubit_dqulearn", "qubit_quclassi"),
        ("DQuLearn qubit vs logistic regression", "qubit_dqulearn", "logistic"),
        (
            "DQuLearn vs QuClassi rule, photonic",
            "photonic_dqulearn",
            "photonic_quclassi",
        ),
        ("Photonic vs qubit, DQuLearn rule", "photonic_dqulearn", "qubit_dqulearn"),
        ("Photonic DQuLearn vs logistic regression", "photonic_dqulearn", "logistic"),
        (
            "Exact shift vs two-point, photonic",
            "photonic_exact_shift",
            "photonic_dqulearn",
        ),
    ):
        comparisons[name] = {
            pair: {
                "points": rows[cand]["pairs"][pair]["mean"]
                - rows[ref]["pairs"][pair]["mean"],
                "ratio": _ratio(by_system[cand][pair], by_system[ref][pair]),
            }
            for pair in PAIRS
        }
    parity = {
        model: all(
            abs(comparisons[name][pair]["points"]) <= PARITY_TOLERANCE_POINTS
            for pair in PAIRS
        )
        for model, name in (
            ("qubit", "DQuLearn vs QuClassi rule, qubit"),
            ("photonic", "DQuLearn vs QuClassi rule, photonic"),
        )
    }
    identity = {}
    for model, sweep in (
        ("qubit", "accuracy_parity"),
        ("photonic", "photonic_accuracy_parity"),
    ):
        parameters = {
            candidate["name"]: np.asarray(metrics["parameters"])
            for candidate, _, metrics in _load(sweeps_dir, sweep)
        }
        identity[model] = max(
            float(np.max(np.abs(values - parameters["local"])))
            for values in parameters.values()
        )
    return {
        "systems": rows,
        "comparisons": comparisons,
        "parity_within_2_points": parity,
        "largest_parameter_difference_local_vs_workers": identity,
        "paper": PAPER["accuracy"],
    }


def worker_scaling(sweeps_dir: Path) -> dict:
    """Epoch seconds per family, workers and layers; 4-vs-1 saving and baseline ratios."""
    cells: dict = {}
    for sweep, qubits in (("worker_scaling_5q", 5), ("worker_scaling_7q", 7)):
        for candidate, seed, metrics in _load(sweeps_dir, sweep):
            labels = candidate["labels"]
            family = labels["family"]
            workers = labels.get("workers", 1)
            key = (qubits, family, workers, labels["layers"])
            cells.setdefault(key, {})[seed] = _finite(
                metrics["epoch_seconds"], candidate["name"]
            )
    table = {}
    for (qubits, family, workers, layers), values in sorted(cells.items()):
        table[f"{qubits}q/{family}/w{workers}/l{layers}"] = _stats(
            list(values.values())
        )
    comparisons = {}
    for qubits in (5, 7):
        manager = "manager_per_circuit"
        for layers in (1, 2, 3):
            four, one = (
                cells[(qubits, manager, 4, layers)],
                cells[(qubits, manager, 1, layers)],
            )
            batched = cells[(qubits, "batched", 1, layers)]
            entry = {
                "4 vs 1 worker": _ratio(four, one),
                "batched (1 process) vs manager with 4 workers": _ratio(batched, four),
            }
            if (qubits, "pool", 4, layers) in cells:
                entry["manager vs process pool, 4 workers"] = _ratio(
                    four, cells[(qubits, "pool", 4, layers)]
                )
            comparisons[f"{qubits}q/l{layers}"] = entry
    return {
        "epoch_seconds": table,
        "comparisons": comparisons,
        "paper_saving_4_vs_1_workers": PAPER["saving_4_vs_1_workers"],
    }


def multi_tenant(sweeps_dir: Path) -> dict:
    """Completion time per job and makespan; multi vs single and policy baselines."""
    readings = {}
    for sweep in ("multi_tenant", "multi_tenant_variants_rerun"):
        per_candidate: dict = {}
        for candidate, seed, metrics in _load(sweeps_dir, sweep):
            values = {
                job: _finite(
                    metrics["jobs"][job]["completion_seconds"], candidate["name"]
                )
                for job in JOBS
            }
            values["makespan"] = _finite(metrics["makespan_seconds"], candidate["name"])
            per_candidate.setdefault(candidate["name"], {})[seed] = values
        readings[sweep] = per_candidate

    def compare(sweep: str, candidate: str, reference: str) -> dict:
        runs = readings[sweep]
        return {
            item: _ratio(
                {s: v[item] for s, v in runs[candidate].items()},
                {s: v[item] for s, v in runs[reference].items()},
            )
            for item in (*JOBS, "makespan")
        }

    seconds = {
        f"{sweep}/{name}": {
            item: _stats([v[item] for v in runs.values()])
            for item in (*JOBS, "makespan")
        }
        for sweep, by_name in readings.items()
        for name, runs in by_name.items()
    }
    return {
        "completion_seconds": seconds,
        "comparisons": {
            "multi vs single tenant (multi_tenant)": compare(
                "multi_tenant", "multi_least_cpu", "single_least_cpu"
            ),
            "multi vs jobs one after another (multi_tenant)": compare(
                "multi_tenant", "multi_least_cpu", "sequential_least_cpu"
            ),
            "least-CPU vs round-robin, multi-tenant (multi_tenant)": compare(
                "multi_tenant", "multi_least_cpu", "multi_round_robin"
            ),
            "multi vs single tenant (rerun, another day)": compare(
                "multi_tenant_variants_rerun", "multi_least_cpu", "single_least_cpu"
            ),
        },
        "paper_change_percent": PAPER["multi_tenant_change"],
    }


def _pm(stat: dict, digits: int = 1) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _ratio_text(ratio: dict) -> str:
    low, high = ratio["ci95"]
    verdict = "differs" if ratio["difference"] else "no difference"
    return f"{ratio['mean']:.3f} [{low:.3f}, {high:.3f}], {verdict}"


def markdown(result: dict) -> str:
    acc, scaling, tenant = (
        result["accuracy"],
        result["worker_scaling"],
        result["multi_tenant"],
    )
    lines = [
        "# Phase 5 comparison",
        "",
        "Built by `utils/build_comparison.py` from the sweeps' structured metrics; "
        "every expected run completed. Local values: one laptop CPU, simulation. "
        "Paper values: its text, never ranked with local values. Rule: a paired "
        "ratio differs when |mean − 1| > 2 std; 95% Student-t interval in brackets.",
        "",
        "## Accuracy kept (claim accuracy-parity)",
        "",
        "Test accuracy (%), mean ± std over seeds 0, 1, 2; 4 PCA features (QuClassi's "
        "preprocessing), 25 epochs, learning rate chosen on validation. Scope: partial.",
        "",
        "| System | Role | Runs | 3/9 | 3/8 | 3/6 | 1/5 | Parameters | Seconds per run |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in acc["systems"].values():
        lines.append(
            f"| {row['name']} | {row['role']} | {row['runs']}/{row['expected_runs']} | "
            + " | ".join(_pm(row["pairs"][pair]) for pair in PAIRS)
            + f" | {row['parameters']} | {row['mean_run_seconds']:.1f} |"
        )
    lines.append(
        "| Paper, DQuLearn with 2 workers | paper-reported | — | "
        + " | ".join(f"{acc['paper'][pair]}" for pair in PAIRS)
        + " | not given | — |"
    )
    lines += [
        "",
        "| Comparison | 3/9 | 3/8 | 3/6 | 1/5 |",
        "|---|---|---|---|---|",
    ]
    for name, per_pair in acc["comparisons"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{per_pair[pair]['points']:+.2f} pts; {_ratio_text(per_pair[pair]['ratio'])}"
                for pair in PAIRS
            )
            + " |"
        )
    identity = acc["largest_parameter_difference_local_vs_workers"]
    parity = acc["parity_within_2_points"]
    lines += [
        "",
        f"Parity within 2 points on every pair: qubit model {parity['qubit']}, "
        f"photonic model {parity['photonic']}. Largest parameter difference between "
        f"local, 1-worker and 2-worker training (3/6, seed 0): qubit {identity['qubit']:g}, "
        f"photonic {identity['photonic']:g}.",
        "",
        "## Epoch time with more workers (claim worker-scaling)",
        "",
        "One-epoch wall-clock (s), mean ± std over 3 repetitions, paper circuit counts, "
        "10,000 shots, per-circuit compilation. Scope: partial (local substitute "
        "environment).",
        "",
        "| Cell (qubits/execution/workers/layers) | Seconds |",
        "|---|---:|",
    ]
    for key, stat in scaling["epoch_seconds"].items():
        lines.append(f"| {key} | {_pm(stat, 2)} |")
    lines += [
        "",
        "| Width and layers | 4 vs 1 worker (ratio) | Paper, 4 vs 1 worker (ratio, Google Cloud) | Manager vs process pool, 4 workers | Batched, 1 process, vs manager with 4 workers |",
        "|---|---|---:|---|---|",
    ]
    for key, entry in scaling["comparisons"].items():
        layers = int(key.split("/l")[1])
        paper = (
            scaling["paper_saving_4_vs_1_workers"][layers]
            if key.startswith("5q")
            else None
        )
        pool = entry.get("manager vs process pool, 4 workers")
        lines.append(
            f"| {key} | {_ratio_text(entry['4 vs 1 worker'])} | "
            + (f"{1 - paper / 100:.3f}" if paper is not None else "not reported")
            + f" | {_ratio_text(pool) if pool else 'not run'} | "
            f"{_ratio_text(entry['batched (1 process) vs manager with 4 workers'])} |"
        )
    lines += [
        "",
        "## Completion time under multi-tenancy (claim multi-tenant-gain)",
        "",
        "Completion time from the common start (s), mean ± std over seeds 0, 1, 2; "
        "workers of 5, 10, 15, 20 qubits, one simulation slot each. Scope: partial "
        "(local substitute environment).",
        "",
        "| Run (sweep/candidate) | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, per_item in tenant["completion_seconds"].items():
        lines.append(
            f"| {key} | "
            + " | ".join(_pm(per_item[item]) for item in (*JOBS, "makespan"))
            + " |"
        )
    lines += [
        "",
        "| Comparison (paired ratio) | 5Q/1L | 5Q/2L | 7Q/1L | 7Q/2L | Makespan |",
        "|---|---|---|---|---|---|",
    ]
    for name, per_item in tenant["comparisons"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(_ratio_text(per_item[item]) for item in (*JOBS, "makespan"))
            + " |"
        )
    paper = tenant["paper_change_percent"]
    lines += [
        "",
        f"Paper (text): 5Q/1L {paper['5Q/1L']}%, 7Q/2L {paper['7Q/2L']}%; bars in "
        "`results/multi_tenant.md`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "accuracy": accuracy(args.sweeps),
        "worker_scaling": worker_scaling(args.sweeps),
        "multi_tenant": multi_tenant(args.sweeps),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.results / "comparison.md").write_text(markdown(result), encoding="utf-8")
    print(f"wrote {args.results / 'comparison.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
