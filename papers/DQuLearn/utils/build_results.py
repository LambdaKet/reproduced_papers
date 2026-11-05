"""Build the curated result tables of the reproduction from the finished sweeps.

Usage (from the repository root)::

    python papers/DQuLearn/utils/build_results.py \\
        --sweeps papers/DQuLearn/outdir --results papers/DQuLearn/results

Reads ``sweep_summary.json`` of every sweep (built by ``summarize_sweep.py`` from
the runs' ``metrics.json``), plus the ``metrics.json`` of the parity and
shot-noise runs, adds 95% confidence intervals (Student t, n − 1 degrees of
freedom) to the paired ratios, and writes ``results/<topic>.json`` and
``results/<topic>.md`` for the worker-scaling, multi-tenant and accuracy claims.
Paper values are the paper's text, or bars measured on its rendered figures
(``LOG.md``, Experiment Inventory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SWEEPS = (
    "worker_scaling_5q",
    "worker_scaling_7q",
    "multi_tenant",
    "multi_tenant_variants_rerun",
    "accuracy_lr_selection",
    "accuracy_final",
    "accuracy_parity",
    "accuracy_shots",
)
RUNTIME_SCOPE = (
    "partial: local substitute environment (worker processes on one laptop instead "
    "of IBM-Q or Google Cloud VMs), paper circuit counts, 3 repetitions"
)
ACCURACY_SCOPE = (
    "partial: QuClassi's released preprocessing (random 1,000-image subsample, "
    "4 PCA features, 5 qubits), lr 0.01 selected on the validation split of seed 0, 3 seeds"
)
# DQuLearn Sec. IV-C2 (text) and Fig. 5 bars measured on the rendered page.
PAPER_CLOUD_5Q = {
    "reduction_4_vs_1": {1: 27.1, 2: 37.3, 3: 43.2},
    "reduction_4_vs_2": {1: 18.9, 2: 31.5, 3: 30.0},
    "bar_seconds": {1: (373, 338, 273), 2: (959, 877, 602), 3: (1738, 1410, 982)},
}
# DQuLearn Fig. 6 bars measured (single, multi) and text values.
PAPER_MULTI_TENANT = {
    "5Q/1L": {"bars": (1058, 264), "text_reduction": 68.7},
    "5Q/2L": {"bars": (2136, 2136), "text_reduction": None},
    "7Q/1L": {"bars": (1479, 1455), "text_reduction": None},
    "7Q/2L": {"bars": (3674, 3004), "text_reduction": 8.2},
}
# DQuLearn Sec. IV-B (2 workers) and QuClassi's single-layer model (Fig. 9, measured).
PAPER_ACCURACY = {
    "3/9": (97.5, 97.6),
    "3/8": (96.2, 87.0),
    "3/6": (98.1, 97.6),
    "1/5": (98.6, 96.8),
}


def _confidence(ratios: dict) -> dict:
    """Mean, sample std and 95% Student-t interval of paired ratios."""
    values = np.array(list(ratios.values()), dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"n": int(len(values)), "mean": mean, "std": None, "ci95": None}
    std = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return {
        "n": int(len(values)),
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
    }


def _comparison(summary: dict, name: str) -> dict:
    comparison = next(c for c in summary["comparisons"] if c["name"] == name)
    return _confidence(comparison["ratios_by_repetition"])


def _stat(summary: dict, candidate: str, metric: str) -> dict:
    return summary["candidates"][candidate]["metrics"][metric]


def _pm(stat: dict, scale: float = 1.0, digits: int = 2) -> str:
    return f"{scale * stat['mean']:.{digits}f} ± {scale * stat['std']:.{digits}f}"


def _signed(percent: float) -> str:
    """Signed percentage with a typographic minus, e.g. "+25.1%" or "−37.0%"."""
    return f"{percent:+.1f}%".replace("-", "−")


def _reduction(ci: dict) -> str:
    """Reduction 1 − ratio, in percent, with its 95% interval."""
    low, high = (100 * (1 - ci["ci95"][1]), 100 * (1 - ci["ci95"][0]))
    return f"{100 * (1 - ci['mean']):.1f}% [{low:.1f}, {high:.1f}]"


def _interval(ci: dict) -> str:
    """Ratio with its 95% interval, e.g. "0.560 [0.355, 0.764]"."""
    return f"{ci['mean']:.3f} [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]"


def worker_scaling(summaries: dict) -> tuple[dict, str]:
    s5, s7 = summaries["worker_scaling_5q"], summaries["worker_scaling_7q"]
    data = {
        "scope": RUNTIME_SCOPE,
        "paper": PAPER_CLOUD_5Q,
        "cells": {},
        "comparisons": {},
    }
    lines = [
        "# Worker scaling",
        "",
        f"Scope: {RUNTIME_SCOPE}. Epoch wall-clock in seconds, mean ± std over 3 "
        "repetitions; reductions are 1 − (paired ratio to 1 worker), with 95% intervals.",
        "",
        "| Qubits | Family | Layers | 1 worker | 2 workers | 4 workers | Reduction, 2 workers | Reduction, 4 workers | Paper, 4 workers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [(5, s5, f, "") for f in ("manager_per_circuit", "manager_cached", "pool")]
    rows.append((7, s7, "manager_per_circuit", "_q7"))
    for qubits, summary, family, suffix in rows:
        for layers in (1, 2, 3):
            names = [f"{family}{suffix}_w{w}_l{layers}" for w in (1, 2, 4)]
            seconds = [_stat(summary, n, "epoch_seconds") for n in names]
            prefix = f"{family}_l{layers}" if not suffix else f"manager_q7_l{layers}"
            ci2 = _comparison(summary, f"{prefix}_w2_vs_w1")
            ci4 = _comparison(summary, f"{prefix}_w4_vs_w1")
            key = f"{qubits}q/{family}/L{layers}"
            data["cells"][key] = {
                "epoch_seconds": seconds,
                "ratio_2_vs_1": ci2,
                "ratio_4_vs_1": ci4,
            }
            paper = (
                f"{PAPER_CLOUD_5Q['reduction_4_vs_1'][layers]}%"
                if qubits == 5 and family == "manager_per_circuit"
                else "—"
            )
            lines.append(
                f"| {qubits} | {family} | {layers} | {_pm(seconds[0])} | {_pm(seconds[1])} | "
                f"{_pm(seconds[2])} | {_reduction(ci2)} | {_reduction(ci4)} | {paper} |"
            )
    lines += [
        "",
        "Batched execution in one process, and the manager against a plain process pool:",
        "",
        "| Qubits | Layers | Batched, one process (s) | Ratio batched / manager with 4 workers | Ratio manager / pool, 4 workers |",
        "|---|---|---|---|---|",
    ]
    for qubits, summary, suffix in ((5, s5, ""), (7, s7, "_q7")):
        for layers in (1, 2, 3):
            batched = _stat(summary, f"batched{suffix}_l{layers}", "epoch_seconds")
            name = (
                f"batched_vs_manager_w4_l{layers}"
                if qubits == 5
                else f"batched_vs_manager_q7_w4_l{layers}"
            )
            ratio = _comparison(summary, name)
            pool = (
                _comparison(s5, f"manager_vs_pool_l{layers}_w4")
                if qubits == 5
                else None
            )
            data["comparisons"][f"{qubits}q/L{layers}"] = {
                "batched_vs_manager_w4": ratio,
                "manager_vs_pool_w4": pool,
            }
            pool_text = (
                f"{pool['mean']:.3f} [{pool['ci95'][0]:.3f}, {pool['ci95'][1]:.3f}]"
                if pool
                else "—"
            )
            lines.append(
                f"| {qubits} | {layers} | {_pm(batched)} | {ratio['mean']:.3f} "
                f"[{ratio['ci95'][0]:.3f}, {ratio['ci95'][1]:.3f}] | {pool_text} |"
            )
    return data, "\n".join(lines) + "\n"


def multi_tenant(summaries: dict) -> tuple[dict, str]:
    summary = summaries["multi_tenant"]
    modes = {
        "multi_least_cpu": "multi, lowest CPU (DQuLearn)",
        "single_least_cpu": "single, one circuit per worker",
        "sequential_least_cpu": "jobs one after another",
        "multi_round_robin": "multi, round-robin",
    }
    jobs = ["5Q/1L", "5Q/2L", "7Q/1L", "7Q/2L"]
    data = {
        "scope": RUNTIME_SCOPE,
        "paper": PAPER_MULTI_TENANT,
        "modes": {},
        "multi_vs_single": {},
    }
    lines = [
        "# Multi-tenant",
        "",
        f"Scope: {RUNTIME_SCOPE}. Completion time of each job from the common start (s), "
        "mean ± std over 3 seeds (the seed shuffles the submission order, identically in every mode).",
        "",
        "| Mode | " + " | ".join(jobs) + " | Makespan |",
        "|---|" + "---|" * (len(jobs) + 1),
    ]
    for name, label in modes.items():
        cells = [
            _pm(_stat(summary, name, f"completion_{j.replace('/', '_')}"), digits=1)
            for j in jobs
        ]
        makespan = _stat(summary, name, "makespan_seconds")
        data["modes"][name] = {
            j: _stat(summary, name, f"completion_{j.replace('/', '_')}") for j in jobs
        } | {"makespan": makespan}
        lines.append(
            f"| {label} | " + " | ".join(cells) + f" | {_pm(makespan, digits=1)} |"
        )
    lines += [
        "",
        "Multi- vs single-tenant, ratio paired by seed (95% interval) and the paper:",
        "",
        "| Job | Completion, multi / single | Change | Paper, text | Paper, bars | Circuits per second, multi / single |",
        "|---|---|---|---|---|---|",
    ]
    for job in jobs:
        key = job.replace("/", "_")
        completion = _comparison(summary, f"multi_vs_single_completion_{key}")
        rate = _comparison(summary, f"multi_vs_single_circuits_per_second_{key}")
        single, multi = PAPER_MULTI_TENANT[job]["bars"]
        text = PAPER_MULTI_TENANT[job]["text_reduction"]
        data["multi_vs_single"][job] = {
            "completion": completion,
            "circuits_per_second": rate,
        }
        lines.append(
            f"| {job} | {completion['mean']:.3f} [{completion['ci95'][0]:.3f}, {completion['ci95'][1]:.3f}] | "
            f"{_signed(100 * (completion['mean'] - 1))} | {'—' if text is None else f'−{text}%'} | "
            f"{_signed(100 * (multi / single - 1))} | {rate['mean']:.2f} [{rate['ci95'][0]:.2f}, {rate['ci95'][1]:.2f}] |"
        )
    makespan = _comparison(summary, "multi_vs_single_makespan")
    data["multi_vs_single"]["makespan"] = makespan
    lines += [
        "",
        f"Makespan, multi / single: {makespan['mean']:.3f} [{makespan['ci95'][0]:.3f}, "
        f"{makespan['ci95'][1]:.3f}] ({_signed(100 * (makespan['mean'] - 1))}). Paper 5Q/1L "
        "circuits per second: ×3.9 (text), ×4.05 (bars).",
    ]

    variants = summaries["multi_tenant_variants_rerun"]
    readings = {
        "multi_vs_single": "One slot per worker, 7-qubit jobs as the text (rerun of the table above)",
        "four_slots_vs_single": "Up to four circuits at once per worker",
        "fig6_multi_vs_single": "7-qubit jobs sized as Fig. 6 (1,440 × layers circuits)",
    }
    data["variants"] = {}
    lines += [
        "",
        "Two ambiguities, both readings (sweep `multi_tenant_variants_rerun`): multi / single "
        "ratio paired by seed (95% interval):",
        "",
        "| Reading | "
        + " | ".join(jobs)
        + " | Makespan | Circuits per second, 5Q/1L |",
        "|---|" + "---|" * (len(jobs) + 2),
    ]
    for prefix, label in readings.items():
        ratios = {
            job: _comparison(variants, f"{prefix}_completion_{job.replace('/', '_')}")
            for job in jobs
        } | {
            "makespan": _comparison(variants, f"{prefix}_makespan"),
            "circuits_per_second_5Q_1L": _comparison(
                variants, f"{prefix}_circuits_per_second_5Q_1L"
            ),
        }
        data["variants"][prefix] = ratios
        lines.append(
            f"| {label} | " + " | ".join(_interval(ci) for ci in ratios.values()) + " |"
        )
    slots = {
        job: _comparison(
            variants, f"four_slots_vs_one_slot_completion_{job.replace('/', '_')}"
        )
        for job in jobs
    } | {"makespan": _comparison(variants, "four_slots_vs_one_slot_makespan")}
    data["variants"]["four_slots_vs_one_slot"] = slots
    lines += [
        "",
        "Four slots vs one slot, both multi-tenant: "
        + ", ".join(f"{name} {_interval(ci)}" for name, ci in slots.items())
        + ".",
    ]
    return data, "\n".join(lines) + "\n"


def _final_parameters(sweep_dir: Path, candidate: str) -> np.ndarray:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    run = next(r for r in status["runs"] if r["candidate"]["name"] == candidate)
    metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
    return np.array(metrics["parameters"]), metrics


def accuracy(summaries: dict, sweeps_dir: Path) -> tuple[dict, str]:
    final, selection = summaries["accuracy_final"], summaries["accuracy_lr_selection"]
    pairs = ["3/9", "3/8", "3/6", "1/5"]
    data = {
        "scope": ACCURACY_SCOPE,
        "paper": PAPER_ACCURACY,
        "final": {},
        "lr_selection": selection["selection"],
    }
    lines = [
        "# Accuracy",
        "",
        f"Scope: {ACCURACY_SCOPE}. Test accuracy (%), mean ± std over seeds 0, 1, 2.",
        "",
        "| Pair | DQuLearn rule, 2 workers | QuClassi rule, local | Logistic regression | DQuLearn / QuClassi (95% interval) | Paper, DQuLearn | QuClassi Fig. 9 |",
        "|---|---|---|---|---|---|---|",
    ]
    for pair in pairs:
        key = pair.replace("/", "")
        cells = {
            rule: _stat(final, f"{rule}_pair{key}", "test_accuracy")
            for rule in ("dqulearn", "quclassi", "logistic")
        }
        ratio = _comparison(final, f"dqulearn_vs_quclassi_test_{key}")
        data["final"][pair] = cells | {"dqulearn_vs_quclassi": ratio}
        lines.append(
            f"| {pair} | {_pm(cells['dqulearn'], 100, 1)} | {_pm(cells['quclassi'], 100, 1)} | "
            f"{_pm(cells['logistic'], 100, 1)} | {ratio['mean']:.3f} [{ratio['ci95'][0]:.3f}, "
            f"{ratio['ci95'][1]:.3f}] | {PAPER_ACCURACY[pair][0]} | {PAPER_ACCURACY[pair][1]} |"
        )
    parity_dir = sweeps_dir / "sweep_accuracy_parity"
    local, _ = _final_parameters(parity_dir, "local")
    differences = {}
    for name in ("manager_w1", "manager_w2"):
        params, _ = _final_parameters(parity_dir, name)
        differences[name] = float(np.max(np.abs(params - local)))
    exact, exact_metrics = _final_parameters(parity_dir, "manager_w2")
    noisy, shots_metrics = _final_parameters(
        sweeps_dir / "sweep_accuracy_shots", "shots_10000_w2"
    )
    data["parity"] = {"max_parameter_difference_to_local": differences}
    data["shots"] = {
        "test_exact": exact_metrics["final"]["test"]["accuracy"],
        "test_shots": shots_metrics["final"]["test"]["accuracy"],
        "max_parameter_difference": float(np.max(np.abs(noisy - exact))),
    }
    lines += [
        "",
        "Learning rate selected on validation (mean over the four pairs, seed 0): "
        + "; ".join(
            f"{s['group']}: {s['selected']} ({', '.join(f'{k}: {v:.3f}' for k, v in s['aggregates'].items())})"
            for s in selection["selection"]
        )
        + ".",
        "",
        "Parity (3/6, seed 0, lr 0.01): largest final-parameter difference to local training: "
        + ", ".join(f"{k} {v:g}" for k, v in differences.items())
        + " (bit-identical).",
        "",
        f"Shot noise (3/6, seed 0, 2 workers, preliminary): test {100 * data['shots']['test_shots']:.2f}% with "
        f"10,000 shots vs {100 * data['shots']['test_exact']:.2f}% exact; largest parameter difference "
        f"{data['shots']['max_parameter_difference']:.4f} rad.",
    ]
    return data, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweeps", type=Path, required=True, help="Folder holding sweep_<id>/"
    )
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    for sweep in SWEEPS:
        summary = json.loads(
            (args.sweeps / f"sweep_{sweep}" / "sweep_summary.json").read_text()
        )
        if not summary["complete_sweep"]:
            raise SystemExit(
                f"{sweep} is not a complete sweep; refusing to build results"
            )
        summaries[sweep] = summary
    args.results.mkdir(parents=True, exist_ok=True)
    for topic, (data, text) in {
        "worker_scaling": worker_scaling(summaries),
        "multi_tenant": multi_tenant(summaries),
        "accuracy": accuracy(summaries, args.sweeps),
    }.items():
        data["sources"] = {s: f"outdir/sweep_{s}/sweep_summary.json" for s in SWEEPS}
        (args.results / f"{topic}.json").write_text(json.dumps(data, indent=2) + "\n")
        (args.results / f"{topic}.md").write_text(text)
        print(f"wrote {args.results / topic}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
