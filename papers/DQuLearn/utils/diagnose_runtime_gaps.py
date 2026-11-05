"""Diagnose the runtime gaps: how much of an epoch runs in parallel, and workers that wait.

Usage (from the repository root)::

    python papers/DQuLearn/utils/diagnose_runtime_gaps.py \\
        --sweeps papers/DQuLearn/outdir --results papers/DQuLearn/results

Worker scaling: fits Amdahl's law, T_w / T_1 = s + (1 − s) / w, to the epoch
time with 4 workers, for the paper (Sec. IV-C2 text; one-worker seconds from the
Fig. 5 bars) and for this reproduction (``sweep_worker_scaling_5q``, DQuLearn
manager). It splits each circuit's one-worker time into a part that does not run
in parallel, s·T_1/n, and a part that does, (1 − s)·T_1/n, and compares the
2-worker time the fit predicts with the measured one. Multi-tenant: seconds per
circuit per worker in the paper (Figs. 5 and 6) and here
(``sweep_worker_scaling_5q``, ``sweep_multi_tenant``), and the multi / single
ratios of ``sweep_multi_tenant_latency`` with and without 0.3 s of latency per
circuit. Writes ``results/runtime_gaps.json`` and ``results/runtime_gaps.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_results import (
    PAPER_CLOUD_5Q,
    PAPER_MULTI_TENANT,
    RUNTIME_SCOPE,
    _comparison,
    _interval,
    _signed,
    _stat,
)

LAYERS = (1, 2, 3)
JOBS = ("5Q/1L", "5Q/2L", "7Q/1L", "7Q/2L")
# Circuits per job in Fig. 6 (runtime × circuits per second of its bars).
FIG6_CIRCUITS = {"5Q/1L": 1440, "5Q/2L": 2880, "7Q/1L": 1440, "7Q/2L": 2880}
LATENCY_SECONDS = 0.3


def _summary(sweeps: Path, sweep: str) -> dict:
    summary = json.loads((sweeps / f"sweep_{sweep}" / "sweep_summary.json").read_text())
    if not summary["complete_sweep"]:
        raise SystemExit(f"{sweep} is not a complete sweep; refusing to use it")
    return summary


def _run_metrics(sweeps: Path, sweep: str) -> list[tuple[str, str, dict]]:
    """(run ID, candidate, metrics) of every run of a sweep."""
    sweep_dir = sweeps / f"sweep_{sweep}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    return [
        (
            run["run_id"],
            run["candidate"]["name"],
            json.loads((sweep_dir / run["metrics_path"]).read_text()),
        )
        for run in status["runs"]
    ]


def overhead_by_workers(sweeps: Path) -> dict:
    """Per-circuit milliseconds of the DQuLearn manager, 5 qubits, by worker count.

    Communication is the round trip seen by the manager minus the execution
    measured on the worker (RPC, queueing and scheduling).
    """
    collected: dict = {}
    for _, name, metrics in _run_metrics(sweeps, "worker_scaling_5q"):
        if not name.startswith("manager_per_circuit_w"):
            continue
        timing = metrics["timing_seconds"]
        workers = name.split("_w")[1].split("_")[0]
        cell = collected.setdefault(workers, {"communication": [], "execution": []})
        cell["communication"].append(
            1000 * (timing["round_trip_mean"] - timing["execution_mean"])
        )
        cell["execution"].append(1000 * timing["execution_mean"])
    return {
        workers: {
            "communication_ms": [
                min(cell["communication"]),
                max(cell["communication"]),
            ],
            "execution_ms": [min(cell["execution"]), max(cell["execution"])],
        }
        for workers, cell in sorted(collected.items())
    }


def seven_qubit_circuits_on_the_5_qubit_worker(sweeps: Path) -> dict:
    """7-qubit circuits that each run, in every tenancy mode, assigned to the 5-qubit worker."""
    counts = {}
    for sweep in ("multi_tenant", "multi_tenant_variants_rerun"):
        for run_id, _, metrics in _run_metrics(sweeps, sweep):
            for worker in metrics["manager"]["workers"]:
                if worker["max_qubits"] == 5:
                    by_client = worker["assigned_by_client"]
                    counts[f"{sweep}/{run_id}"] = by_client.get(
                        "7Q/1L", 0
                    ) + by_client.get("7Q/2L", 0)
    return counts


def _amdahl(
    ratio_4: float, ratio_2: float, one_worker_seconds: float, circuits: int
) -> dict:
    """Fit s to the 4-worker ratio and split the one-worker time per circuit."""
    serial = (4 * ratio_4 - 1) / 3
    return {
        "ratio_4_vs_1": ratio_4,
        "ratio_2_vs_1": ratio_2,
        "serial_fraction": serial,
        "serial_seconds_per_circuit": serial * one_worker_seconds / circuits,
        "parallel_seconds_per_circuit": (1 - serial) * one_worker_seconds / circuits,
        "predicted_ratio_2_vs_1": serial + (1 - serial) / 2,
    }


def worker_scaling(scaling: dict) -> dict:
    fits = {}
    for layers in LAYERS:
        circuits = 1440 * layers
        paper_4 = 1 - PAPER_CLOUD_5Q["reduction_4_vs_1"][layers] / 100
        paper_2 = paper_4 / (1 - PAPER_CLOUD_5Q["reduction_4_vs_2"][layers] / 100)
        prefix = f"manager_per_circuit_l{layers}"
        fits[f"L{layers}"] = {
            "paper": _amdahl(
                paper_4, paper_2, PAPER_CLOUD_5Q["bar_seconds"][layers][0], circuits
            ),
            "reproduction": _amdahl(
                _comparison(scaling, f"{prefix}_w4_vs_w1")["mean"],
                _comparison(scaling, f"{prefix}_w2_vs_w1")["mean"],
                _stat(scaling, f"manager_per_circuit_w1_l{layers}", "epoch_seconds")[
                    "mean"
                ],
                circuits,
            ),
        }
    return fits


def seconds_per_circuit(scaling: dict, tenant: dict) -> dict:
    """Seconds per circuit per worker; four single-tenant jobs assumed to run together."""
    paper_single_rate = sum(
        FIG6_CIRCUITS[job] / PAPER_MULTI_TENANT[job]["bars"][0] for job in JOBS
    )
    our_single_rate = sum(
        _stat(
            tenant, "single_least_cpu", f"circuits_per_second_{job.replace('/', '_')}"
        )["mean"]
        for job in JOBS
    )
    one_worker = _stat(scaling, "manager_per_circuit_w1_l1", "epoch_seconds")["mean"]
    return {
        "one_worker_5Q_1L": {
            "paper": PAPER_CLOUD_5Q["bar_seconds"][1][0] / 1440,
            "reproduction": one_worker / 1440,
        },
        "single_tenant_four_jobs": {
            "paper": 4 / paper_single_rate,
            "reproduction": 4 / our_single_rate,
        },
    }


def latency(sweep: dict) -> dict:
    regimes = {}
    for regime in ("no_latency", "latency"):
        ratios = {
            job: _comparison(
                sweep, f"{regime}_multi_vs_single_completion_{job.replace('/', '_')}"
            )
            for job in JOBS
        }
        ratios["makespan"] = _comparison(sweep, f"{regime}_multi_vs_single_makespan")
        ratios["circuits_per_second_5Q_1L"] = _comparison(
            sweep, f"{regime}_multi_vs_single_circuits_per_second_5Q_1L"
        )
        regimes[regime] = ratios
    return regimes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweeps", type=Path, required=True, help="Folder holding sweep_<id>/"
    )
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    scaling = _summary(args.sweeps, "worker_scaling_5q")
    tenant = _summary(args.sweeps, "multi_tenant")
    latency_sweep = _summary(args.sweeps, "multi_tenant_latency")
    data = {
        "scope": RUNTIME_SCOPE,
        "worker_scaling_amdahl": worker_scaling(scaling),
        "seconds_per_circuit_per_worker": seconds_per_circuit(scaling, tenant),
        "multi_vs_single_with_latency": latency(latency_sweep),
        "latency_seconds": LATENCY_SECONDS,
        "overhead_by_workers": overhead_by_workers(args.sweeps),
        "seven_qubit_circuits_on_the_5_qubit_worker": seven_qubit_circuits_on_the_5_qubit_worker(
            args.sweeps
        ),
    }

    lines = [
        "# Runtime gaps",
        "",
        f"Scope: {RUNTIME_SCOPE}. Amdahl fit T_w / T_1 = s + (1 − s) / w on the 4-worker time; "
        "s is the share of the one-worker epoch that does not run in parallel.",
        "",
        "| Layers | Source | 4 workers / 1 | Not parallel, s | Not parallel, s per circuit | "
        "Parallel, s per circuit | 2 workers / 1, measured | 2 workers / 1, fit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for layers, fit in data["worker_scaling_amdahl"].items():
        for source in ("paper", "reproduction"):
            f = fit[source]
            lines.append(
                f"| {layers[1:]} | {source} | {f['ratio_4_vs_1']:.3f} | {f['serial_fraction']:.2f} | "
                f"{f['serial_seconds_per_circuit']:.4f} | {f['parallel_seconds_per_circuit']:.4f} | "
                f"{f['ratio_2_vs_1']:.3f} | {f['predicted_ratio_2_vs_1']:.3f} |"
            )
    per_circuit = data["seconds_per_circuit_per_worker"]
    lines += [
        "",
        "Seconds per circuit per worker (single-tenant: four jobs assumed to run together; "
        "paper from the Fig. 5 and Fig. 6 bars):",
        "",
        "| Setting | Paper | This reproduction | Ratio |",
        "|---|---|---|---|",
    ]
    for name, label in (
        ("one_worker_5Q_1L", "One worker, 5Q/1L epoch"),
        ("single_tenant_four_jobs", "Single-tenant, four jobs on four workers"),
    ):
        entry = per_circuit[name]
        lines.append(
            f"| {label} | {entry['paper']:.3f} | {entry['reproduction']:.4f} | "
            f"×{entry['paper'] / entry['reproduction']:.0f} |"
        )
    lines += [
        "",
        f"Multi / single ratio paired by seed (95% interval), workers without latency and with "
        f"{LATENCY_SECONDS} s per circuit (jobs at one tenth of Fig. 6's sizes), and the paper:",
        "",
        "| Workers | "
        + " | ".join(JOBS)
        + " | Makespan | Circuits per second, 5Q/1L |",
        "|---|" + "---|" * (len(JOBS) + 2),
    ]
    for regime, label in (
        ("no_latency", "Compute only"),
        ("latency", f"{LATENCY_SECONDS} s latency per circuit"),
    ):
        ratios = data["multi_vs_single_with_latency"][regime]
        lines.append(
            f"| {label} | " + " | ".join(_interval(ci) for ci in ratios.values()) + " |"
        )
    text = [PAPER_MULTI_TENANT[job]["text_reduction"] for job in JOBS]
    bars = [PAPER_MULTI_TENANT[job]["bars"] for job in JOBS]
    makespan_bars = max(multi for _, multi in bars) / max(single for single, _ in bars)
    lines += [
        "| Paper, text | "
        + " | ".join(
            "—" if value is None else f"{1 - value / 100:.3f}" for value in text
        )
        + " | — | 3.9 |",
        "| Paper, Fig. 6 bars | "
        + " | ".join(f"{multi / single:.3f}" for single, multi in bars)
        + f" | {makespan_bars:.3f} (longest job) | 4.05 |",
        "",
        "Changes: "
        + "; ".join(
            f"{label} 5Q/1L {_signed(100 * (ratios['5Q/1L']['mean'] - 1))}, makespan "
            f"{_signed(100 * (ratios['makespan']['mean'] - 1))}"
            for label, ratios in (
                ("compute only", data["multi_vs_single_with_latency"]["no_latency"]),
                ("with latency", data["multi_vs_single_with_latency"]["latency"]),
            )
        )
        + ".",
        "",
        "Per-circuit milliseconds of the DQuLearn manager, 5 qubits, 1–3 layers (range of the "
        "nine runs per worker count); communication is the manager's round trip minus the "
        "worker's execution:",
        "",
        "| Workers | Communication | Execution |",
        "|---|---|---|",
    ]
    for workers, cell in data["overhead_by_workers"].items():
        communication, execution = cell["communication_ms"], cell["execution_ms"]
        lines.append(
            f"| {workers} | {communication[0]:.2f}–{communication[1]:.2f} | "
            f"{execution[0]:.1f}–{execution[1]:.1f} |"
        )
    seven_qubit = data["seven_qubit_circuits_on_the_5_qubit_worker"]
    lines += [
        "",
        f"7-qubit circuits assigned to the 5-qubit worker: {sum(seven_qubit.values())} "
        f"over the {len(seven_qubit)} runs of `multi_tenant` and "
        "`multi_tenant_variants_rerun`, every tenancy mode included.",
    ]
    data["sources"] = {
        sweep: f"outdir/sweep_{sweep}/sweep_summary.json"
        for sweep in (
            "worker_scaling_5q",
            "multi_tenant",
            "multi_tenant_variants_rerun",
            "multi_tenant_latency",
        )
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "runtime_gaps.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.results / "runtime_gaps.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.results / 'runtime_gaps'}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
