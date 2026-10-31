"""Print the result table of a sweep from its structured artifacts.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/sweep_table.py <sweep_dir>

Reads ``sweep_spec.json``, ``sweep_status.json`` and each completed run's
``metrics.json`` and ``run_status.json``; never reads logs. Checks that the
observed runs match the planned candidates (no missing, duplicate, failed or
non-finite result, one code commit) and prints a Markdown table with full
precision kept in the JSON written next to the sweep (``sweep_table.json``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _row(sweep_dir: Path, run: dict) -> dict:
    row = {
        "candidate": run["run_id"],
        "args": " ".join(run["candidate"].get("args", [])),
    }
    row["status"] = run["status"]
    if run["status"] != "COMPLETED":
        return row
    metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
    run_status = json.loads((sweep_dir / run["run_status"]).read_text())
    row["commit"] = run_status["code"]["commit"]
    row["dirty"] = run_status["code"]["dirty"]
    row["kind"] = metrics["kind"]
    if metrics["kind"] == "qgan":
        row["n"] = metrics["final_fidelity"]["n"]
        row["final_mean"] = metrics["final_fidelity"]["mean"]
        row["final_std"] = metrics["final_fidelity"]["std"]
        row["noise_averaged_mean"] = metrics["noise_averaged_final_fidelity"]["mean"]
        row["first_reaches_0p9"] = metrics["mean_fidelity_first_reaches"]["round"]
        row["convergence_mean"] = metrics["convergence_round"]["mean"]
        row["convergence_std"] = metrics["convergence_round"]["std"]
        row["mean_curve"] = metrics["mean_fidelity_by_round"]
        values = [row["final_mean"], row["final_std"], row["noise_averaged_mean"]]
    else:
        row["n"] = metrics["best_fidelity"]["n"]
        row["final_mean"] = metrics["best_fidelity"]["mean"]
        row["final_std"] = metrics["best_fidelity"]["std"]
        row["noise_averaged"] = {
            k: v["mean"] for k, v in metrics["noise_averaged_best_fidelity"].items()
        }
        values = [row["final_mean"], row["final_std"], *row["noise_averaged"].values()]
    row["finite"] = all(math.isfinite(v) for v in values)
    return row


def build(sweep_dir: Path) -> dict:
    """Return the checked table of a sweep as a dict."""
    spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    planned = [c["name"] for c in spec["candidates"]]
    observed = [r["run_id"] for r in status["runs"]]
    rows = [_row(sweep_dir, run) for run in status["runs"]]
    completed = [r for r in rows if r["status"] == "COMPLETED"]
    problems = []
    if sorted(planned) != sorted(observed) or len(set(observed)) != len(observed):
        problems.append("observed candidates differ from the plan or repeat")
    problems += [
        f"{r['candidate']}: {r['status']}" for r in rows if r["status"] != "COMPLETED"
    ]
    problems += [
        f"{r['candidate']}: non-finite metric" for r in completed if not r["finite"]
    ]
    commits = {(r["commit"], r["dirty"]) for r in completed}
    if len(commits) > 1:
        problems.append(f"runs come from several code states: {sorted(commits)}")
    return {
        "sweep_id": spec["sweep_id"],
        "sweep_status": status["status"],
        "expected_runs": len(planned),
        "completed_runs": len(completed),
        "coverage": "COMPLETE SWEEP" if not problems else "PARTIAL SWEEP",
        "problems": problems,
        "code_states": sorted(commits),
        "rows": rows,
    }


def markdown(table: dict) -> str:
    """Markdown rendering of ``build``'s output (3-decimal display only)."""
    lines = [
        f"{table['sweep_id']}: {table['coverage']}, {table['completed_runs']}/"
        f"{table['expected_runs']} runs; code {table['code_states']}",
        "",
    ]
    qgan = any(r.get("kind") == "qgan" for r in table["rows"])
    if qgan:
        lines += [
            "| Candidate | Targets | Final fidelity (mean ± std) | Noise-averaged final | "
            "Round mean reaches 0.9 | Convergence round (mean ± std) | Status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in table["rows"]:
            if r["status"] != "COMPLETED":
                lines.append(f"| {r['candidate']} | | | | | | {r['status']} |")
                continue
            lines.append(
                f"| {r['candidate']} | {r['n']} | {r['final_mean']:.3f} ± {r['final_std']:.3f} "
                f"| {r['noise_averaged_mean']:.3f} | {r['first_reaches_0p9']} "
                f"| {r['convergence_mean']:.0f} ± {r['convergence_std']:.0f} | COMPLETED |"
            )
    else:
        sigmas = next(
            iter(r["noise_averaged"] for r in table["rows"] if "noise_averaged" in r)
        )
        lines += [
            "| Candidate | Targets | Best fidelity (mean ± std) | "
            + " | ".join(f"Noise-averaged σ = {s}π" for s in sigmas)
            + " | Status |",
            "|---|---:|---:|" + "---:|" * len(sigmas) + "---|",
        ]
        for r in table["rows"]:
            if r["status"] != "COMPLETED":
                lines.append(
                    f"| {r['candidate']} | | | {'| ' * len(sigmas)}{r['status']} |"
                )
                continue
            cells = " | ".join(f"{v:.3f}" for v in r["noise_averaged"].values())
            lines.append(
                f"| {r['candidate']} | {r['n']} | {r['final_mean']:.3f} ± {r['final_std']:.3f} "
                f"| {cells} | COMPLETED |"
            )
    if table["problems"]:
        lines += ["", "Problems: " + "; ".join(table["problems"])]
    return "\n".join(lines)


def main(sweep_dir: Path) -> int:
    table = build(sweep_dir)
    compact = {
        **table,
        "rows": [
            {k: v for k, v in r.items() if k != "mean_curve"} for r in table["rows"]
        ],
    }
    (sweep_dir / "sweep_table.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(markdown(table))
    return 0 if table["coverage"] == "COMPLETE SWEEP" else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
