"""Summarize a finished DQuLearn sweep from its runs' ``metrics.json`` files.

Usage (from the repository root)::

    python papers/DQuLearn/utils/summarize_sweep.py <sweep_dir>

Reads ``sweep_spec.json`` and ``sweep_status.json`` in the sweep folder, loads
the ``metrics.json`` of every COMPLETED run, checks coverage against the planned
candidate-by-repetition set, rejects missing or non-finite values, and writes
``sweep_summary.json`` and ``sweep_summary.md`` next to them:

- per candidate and declared metric: completed and expected runs, mean,
  standard deviation (ddof = 1), median, minimum, maximum and the values by
  repetition;
- each declared comparison: the ratio candidate / reference of a metric, paired
  by repetition, its mean and standard deviation, and the reduction 1 − ratio;
- the optional selection of the spec, with its rule and tie tolerance.

Values are read at full precision from the structured artifacts; no log is parsed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _extract(metrics: dict, paths: list) -> float | None:
    """First value found along the alternative paths (dict keys or list indices), or None."""
    for path in paths:
        value = metrics
        for key in path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif (
                isinstance(value, list)
                and isinstance(key, int)
                and -len(value) <= key < len(value)
            ):
                value = value[key]
            else:
                value = None
                break
        if value is not None:
            return float(value)
    return None


def _stats(values_by_repetition: dict[int, float], expected: int) -> dict:
    values = np.array(list(values_by_repetition.values()), dtype=float)
    return {
        "n": int(len(values)),
        "expected": expected,
        "mean": float(values.mean()) if len(values) else None,
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(np.median(values)) if len(values) else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "by_repetition": {str(k): v for k, v in sorted(values_by_repetition.items())},
    }


def _select(
    spec_selection: dict, candidates: dict, complete_names: set[str]
) -> list[dict]:
    """Apply the spec's selection rule within each group."""
    metric = spec_selection["metric"]
    group_key, choose_key = spec_selection["group_by"], spec_selection["choose_by"]
    maximize = spec_selection["direction"] == "maximize"
    tolerance = float(spec_selection["tie_tolerance"])
    results = []
    groups = sorted({c["labels"][group_key] for c in candidates.values()})
    for group in groups:
        members = {
            n: c for n, c in candidates.items() if c["labels"][group_key] == group
        }
        options: dict[float, list[float]] = {}
        partial = False
        for name, candidate in members.items():
            mean = candidate["metrics"][metric]["mean"]
            if name not in complete_names or mean is None:
                partial = True
                continue
            options.setdefault(float(candidate["labels"][choose_key]), []).append(mean)
        aggregates = {
            option: float(np.mean(values)) for option, values in options.items()
        }
        if not aggregates:
            continue
        best = max(aggregates.values()) if maximize else min(aggregates.values())
        tied = sorted(o for o, v in aggregates.items() if abs(v - best) <= tolerance)
        chosen = tied[0] if spec_selection["tie_rule"] == "smaller" else tied[-1]
        results.append(
            {
                "group": group,
                "selected": {choose_key: chosen},
                "tied": tied,
                "aggregates": {str(k): v for k, v in sorted(aggregates.items())},
                "status": "partial" if partial else "complete",
                "rule": (
                    f"{'highest' if maximize else 'lowest'} mean {metric} over "
                    f"{spec_selection['aggregate_over']}; ties within {tolerance} go to the "
                    f"{spec_selection['tie_rule']} {choose_key}"
                ),
            }
        )
    return results


def summarize(sweep_dir: Path) -> dict:
    spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    metric_paths = spec["metrics"]
    expected_per_candidate = len(spec["seeds"])
    values: dict[str, dict[str, dict[int, float]]] = {
        c["name"]: {m: {} for m in metric_paths} for c in spec["candidates"]
    }
    missing, invalid, commits = [], [], set()
    for run in status["runs"]:
        name, repetition = run["candidate"]["name"], int(run["repetition"])
        if run["status"] != "COMPLETED" or not run.get("metrics_path"):
            missing.append(run["run_id"])
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        run_status = json.loads((sweep_dir / run["run_status"]).read_text())
        commits.add((run_status["code"]["commit"], run_status["code"]["dirty"]))
        extracted = {m: _extract(metrics, paths) for m, paths in metric_paths.items()}
        bad = [m for m, v in extracted.items() if v is None or not math.isfinite(v)]
        if bad:
            invalid.append({"run_id": run["run_id"], "metrics": bad})
            continue
        for metric, value in extracted.items():
            if repetition in values[name][metric]:
                invalid.append(
                    {"run_id": run["run_id"], "reason": "duplicate repetition"}
                )
            values[name][metric][repetition] = value

    candidates = {}
    complete_names = set()
    for candidate in spec["candidates"]:
        name = candidate["name"]
        stats = {
            m: _stats(values[name][m], expected_per_candidate) for m in metric_paths
        }
        if all(s["n"] == expected_per_candidate for s in stats.values()):
            complete_names.add(name)
        candidates[name] = {"labels": candidate.get("labels", {}), "metrics": stats}

    comparisons = []
    for comparison in spec.get("comparisons", []):
        metric = comparison["metric"]
        numerator = values[comparison["candidate"]][metric]
        denominator = values[comparison["reference"]][metric]
        paired = sorted(set(numerator) & set(denominator))
        ratios = np.array([numerator[r] / denominator[r] for r in paired], dtype=float)
        comparisons.append(
            {
                "name": comparison["name"],
                "metric": metric,
                "candidate": comparison["candidate"],
                "reference": comparison["reference"],
                "pairs": len(paired),
                "ratio_mean": float(ratios.mean()) if len(ratios) else None,
                "ratio_std": float(ratios.std(ddof=1)) if len(ratios) > 1 else None,
                "reduction_mean": float(1.0 - ratios.mean()) if len(ratios) else None,
                "ratios_by_repetition": dict(zip(map(str, paired), ratios.tolist())),
            }
        )

    selection = (
        _select(spec["selection"], candidates, complete_names)
        if spec.get("selection")
        else []
    )
    complete = not missing and not invalid and len(commits) == 1
    return {
        "sweep_id": spec["sweep_id"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coverage": {
            "expected_runs": status["expected_runs"],
            "completed_runs": status["expected_runs"] - len(missing) - len(invalid),
            "missing": missing,
            "invalid": invalid,
            "code_states": [{"commit": c, "dirty": d} for c, d in sorted(commits)],
        },
        "complete_sweep": complete,
        "label": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "candidates": candidates,
        "comparisons": comparisons,
        "selection": selection,
        "selected": [
            {
                "group": s["group"],
                **s["selected"],
                "rule": s["rule"],
                "status": s["status"],
            }
            for s in selection
        ],
    }


def _fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def to_markdown(summary: dict, metric_order: list[str]) -> str:
    lines = [
        f"# Sweep `{summary['sweep_id']}` — {summary['label']}",
        "",
        f"Generated {summary['generated_at']} from the runs' metrics.json files. Coverage: "
        f"{summary['coverage']['completed_runs']}/{summary['coverage']['expected_runs']} runs; "
        f"code states {summary['coverage']['code_states']}.",
        "",
    ]
    for metric in metric_order:
        lines += [
            f"## {metric}",
            "",
            "| Candidate | Runs | Mean | Std | Median | Min | Max |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, candidate in summary["candidates"].items():
            s = candidate["metrics"][metric]
            lines.append(
                f"| {name} | {s['n']}/{s['expected']} | {_fmt(s['mean'])} | {_fmt(s['std'])} | "
                f"{_fmt(s['median'])} | {_fmt(s['min'])} | {_fmt(s['max'])} |"
            )
        lines.append("")
    if summary["comparisons"]:
        lines += [
            "## Comparisons (ratio candidate / reference, paired by repetition)",
            "",
            "| Comparison | Metric | Pairs | Ratio mean | Ratio std | Reduction |",
            "|---|---|---|---|---|---|",
        ]
        for c in summary["comparisons"]:
            reduction = (
                None if c["reduction_mean"] is None else 100 * c["reduction_mean"]
            )
            lines.append(
                f"| {c['name']} | {c['metric']} | {c['pairs']} | {_fmt(c['ratio_mean'])} | "
                f"{_fmt(c['ratio_std'])} | {_fmt(reduction, 3)}% |"
            )
        lines.append("")
    for s in summary["selection"]:
        lines += [
            f"Selection ({s['status']}), group `{s['group']}`: {s['selected']} — {s['rule']}; "
            f"aggregates {s['aggregates']}; tied {s['tied']}.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    summary = summarize(sweep_dir)
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
    (sweep_dir / "sweep_summary.md").write_text(
        to_markdown(summary, list(spec["metrics"]))
    )
    print(
        f"{summary['label']}: {summary['coverage']['completed_runs']}/"
        f"{summary['coverage']['expected_runs']} runs; summary in {sweep_dir}"
    )
    return 0 if summary["coverage"]["completed_runs"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
