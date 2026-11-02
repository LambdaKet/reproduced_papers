"""Compare the QGAN with its matched baselines (phase 05).

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/compare_baselines.py \\
        papers/qgan_quantum_data_photonics/outdir \\
        --out papers/qgan_quantum_data_photonics/results

Reads the ``metrics.json``, ``run_status.json`` and ``trajectories.npz`` files
listed by the ``sweep_status.json`` of the sweeps below, never logs. Every sweep
must be COMPLETED with every expected run, and every run must cover target
seeds 0-99 with finite per-target values. Applies the decision rule set before
these comparisons, with the 0.02 tolerance of the earlier ceiling and MerLin
rules (root fidelity):

- tied: |mean paired difference| <= 0.02;
- clearly higher or lower: |mean| > 0.02 and the 95 % interval excludes 0;
- otherwise no clear separation.

Paired differences use the same target seeds in both arms (the inputs of a seed
do not depend on the condition); 95 % percentile bootstrap over targets, 10,000
resamples, generator seed 0. Writes ``baseline_comparison.json`` and ``.md`` to
``--out`` and prints the Markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import paper_values as paper

SEEDS = list(range(100))
SIGMAS = (0.01, 0.02, 0.03, 0.04, 0.05)
STUCK = tuple(range(0, 25, 2))
MERLIN_STUCK = (0, 8, 14, 16, 20)
COUNTS = (100, 200, 300, 400, 500, 1000, 1500, 2000)
TOLERANCE = 0.02
RESAMPLES = 10_000
DEFECT_SWEEPS = {
    "defect_rate_sweep_low": (0, 2, 4, 6, 8),
    "defect_rate_sweep_mid": (10, 12, 14, 16),
    "defect_rate_sweep_high": (18, 20, 22, 24),
}
EXPECTED_RUNS = {
    "phase_noise_sweep": [f"sigma_{s}pi" for s in SIGMAS],
    **{
        sweep: [f"stuck_{k}_of_30" for k in stuck]
        for sweep, stuck in DEFECT_SWEEPS.items()
    },
    "shot_noise_sweep": [f"counts_{n}" for n in COUNTS],
    "single_runs": ["single_defect", "fixed_discriminator"],
    "noise_free_reference": ["noise_free_400_rounds", "noise_free_2500_rounds"],
    "oracle_ceiling": [f"ceiling_stuck_{k}_of_30" for k in STUCK],
    "merlin_native_phase_noise": [f"merlin_phase_error_sigma_{s}pi" for s in SIGMAS],
    "merlin_native_defects": [
        f"merlin_phase_error_stuck_{k}_of_30" for k in MERLIN_STUCK
    ],
}
# Systems: the runs each one contributes, its role and what it is matched on.
SYSTEMS = {
    "QGAN, our reproduction": {
        "runs": [
            ("phase_noise_sweep", None),
            *((sweep, None) for sweep in DEFECT_SWEEPS),
            ("shot_noise_sweep", None),
            ("single_runs", "single_defect"),
        ],
        "role": "reproduced method (numpy chip simulator)",
        "provenance": "local runs, phase 03",
        "resources": "30 generator + 12 discriminator phases; parameter shift; "
        "1500 counts per setting (Suppl. Fig. 9b: 100-2000)",
        "status": "reference",
    },
    "Noise-free QGAN": {
        "runs": [("noise_free_reference", None)],
        "role": "baseline: cost of noise (phase-01 plan)",
        "provenance": "local runs, phase 05",
        "resources": "same phases and settings; sigma = 0, exact probabilities",
        "status": "COMMENSURABLE",
    },
    "Oracle ceiling": {
        "runs": [("oracle_ceiling", None)],
        "role": "control: best fidelity the chip can reach (knows the target)",
        "provenance": "local runs, phase 03",
        "resources": "30 generator phases minus stuck ones; exact gradients; "
        "20 restarts x 2000 Adam iterations; 1000 noise draws per sigma",
        "status": "COMMENSURABLE (upper bound by design)",
    },
    "Fixed discriminator": {
        "runs": [("single_runs", "fixed_discriminator")],
        "role": "ablation: random projectors redrawn every round",
        "provenance": "local run, phase 03",
        "resources": "30 generator phases; no discriminator training",
        "status": "COMMENSURABLE",
    },
    "MerLin adaptation, MerLin noise": {
        "runs": [("merlin_native_phase_noise", None), ("merlin_native_defects", None)],
        "role": "photonic adaptation (MerLin QuantumLayer, 8 modes, 2 photons)",
        "provenance": "local runs, phase 04 (simulation)",
        "resources": "same as the QGAN; MerLin phase_error noise "
        "(uniform, same standard deviation)",
        "status": "PARTIALLY_COMMENSURABLE (noise distribution; target "
        "prepared on the chip)",
    },
}
# Runs at once on the laptop CPU, one thread each (LOG: Sweep Records).
CONCURRENCY = {
    "local runs, phase 03": "up to 10 runs at once",
    "local run, phase 03": "up to 10 runs at once",
    "local runs, phase 04 (simulation)": "3 runs at once",
    "local runs, phase 05": "1 run at a time",
}


def _load(outdir: Path) -> dict[tuple[str, str], tuple[dict, Path]]:
    """Every expected run, checked; keyed by (sweep, run id)."""
    runs = {}
    for sweep, expected in EXPECTED_RUNS.items():
        sweep_dir = outdir / f"sweep_{sweep}"
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(f"{sweep} is {status['status']}, not COMPLETED")
        ids = [run["run_id"] for run in status["runs"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{sweep}: duplicate run ids {ids}")
        if missing := sorted(set(expected) - set(ids)):
            raise ValueError(f"{sweep}: missing runs {missing}")
        for run in status["runs"]:
            if run["run_id"] not in expected:
                continue
            run_dir = sweep_dir / run["run_dir"]
            run_status = json.loads((run_dir / "run_status.json").read_text())
            if run["status"] != "COMPLETED" or run_status["status"] != "COMPLETED":
                raise ValueError(f"{sweep}/{run['run_id']} is not COMPLETED")
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            if metrics["seeds"] != SEEDS:
                raise ValueError(f"{sweep}/{run['run_id']}: seeds are not 0-99")
            runs[sweep, run["run_id"]] = (metrics, run_dir)
    return runs


def _values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(SEEDS),) or not np.isfinite(array).all():
        raise ValueError(f"expected {len(SEEDS)} finite values, got {array.shape}")
    return array


def _final(runs, sweep: str, run_id: str) -> np.ndarray:
    return _values(runs[sweep, run_id][0]["final_fidelity_per_seed"])


def _noise_averaged_last_round(runs, sweep: str, run_id: str) -> np.ndarray:
    metrics, run_dir = runs[sweep, run_id]
    arrays = np.load(run_dir / metrics["artifacts"]["trajectories"])
    if int(arrays["noise_averaged_rounds"][-1]) != metrics["settings"]["rounds"]:
        raise ValueError(f"{sweep}/{run_id}: no noise average at the last round")
    return _values(arrays["noise_averaged"][-1])


def _defect_run(stuck: int) -> tuple[str, str]:
    sweep = next(s for s, ks in DEFECT_SWEEPS.items() if stuck in ks)
    return sweep, f"stuck_{stuck}_of_30"


def _compare(values: np.ndarray, reference: np.ndarray, **fields) -> dict:
    """Paired difference values - reference (system - control) and the rule."""
    diff = values - reference
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, diff.size, (RESAMPLES, diff.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    mean = float(diff.mean())
    if abs(mean) <= TOLERANCE:
        outcome = "tied"
    elif low > 0 or high < 0:
        outcome = "clearly higher" if mean > 0 else "clearly lower"
    else:
        outcome = "no clear separation"
    return {
        **fields,
        "system_mean": float(values.mean()),
        "system_std": float(values.std(ddof=1)),
        "control_mean": float(reference.mean()),
        "control_std": float(reference.std(ddof=1)),
        "difference": {"mean": mean, "ci95": [float(low), float(high)]},
        "n": int(diff.size),
        "outcome": outcome,
    }


def comparisons(runs) -> list[dict]:
    rows = []
    noise_free_400 = _final(runs, "noise_free_reference", "noise_free_400_rounds")
    ceiling_intact = runs["oracle_ceiling", "ceiling_stuck_0_of_30"][0]
    for sigma in SIGMAS:
        condition = f"sigma = {sigma}pi"
        qgan = _final(runs, "phase_noise_sweep", f"sigma_{sigma}pi")
        rows.append(
            _compare(
                qgan,
                noise_free_400,
                claim="phase-noise-robustness",
                condition=condition,
                system="QGAN",
                control="Noise-free QGAN",
                metric="final root fidelity, intended phases",
                status="COMMENSURABLE",
                paper=paper.PHASE_NOISE_FINAL[sigma],
            )
        )
        rows.append(
            _compare(
                _noise_averaged_last_round(
                    runs, "phase_noise_sweep", f"sigma_{sigma}pi"
                ),
                _values(
                    ceiling_intact["noise_averaged_best_fidelity_per_seed"][str(sigma)]
                ),
                claim="phase-noise-robustness",
                condition=condition,
                system="QGAN",
                control="Oracle ceiling",
                metric="noise-averaged root fidelity (QGAN at round 400, "
                "100 draws; ceiling at its optimum, 1000 draws)",
                status="COMMENSURABLE",
            )
        )
        rows.append(
            _compare(
                _final(
                    runs,
                    "merlin_native_phase_noise",
                    f"merlin_phase_error_sigma_{sigma}pi",
                ),
                qgan,
                claim="phase-noise-robustness",
                condition=condition,
                system="MerLin adaptation, MerLin noise",
                control="QGAN",
                metric="final root fidelity, intended phases",
                status="PARTIALLY_COMMENSURABLE",
            )
        )
    for stuck, paper_value in zip(
        paper.DEFECT_STUCK, paper.DEFECT_FIDELITY, strict=True
    ):
        condition = f"{stuck}/30 stuck"
        ceiling = _values(
            runs["oracle_ceiling", f"ceiling_stuck_{stuck}_of_30"][0][
                "best_fidelity_per_seed"
            ]
        )
        rows.append(
            _compare(
                _final(runs, *_defect_run(stuck)),
                ceiling,
                claim="defect-tolerance",
                condition=condition,
                system="QGAN",
                control="Oracle ceiling",
                metric="final root fidelity, intended phases (ceiling: best)",
                status="COMMENSURABLE",
                paper=paper_value,
            )
        )
        if stuck in MERLIN_STUCK:
            rows.append(
                _compare(
                    _final(
                        runs,
                        "merlin_native_defects",
                        f"merlin_phase_error_stuck_{stuck}_of_30",
                    ),
                    ceiling,
                    claim="defect-tolerance",
                    condition=condition,
                    system="MerLin adaptation, MerLin noise",
                    control="Oracle ceiling",
                    metric="final root fidelity, intended phases (ceiling: best)",
                    status="PARTIALLY_COMMENSURABLE",
                )
            )
    rows.append(
        _compare(
            _final(runs, "single_runs", "single_defect"),
            _final(runs, *_defect_run(0)),
            claim="no-crucial-phase-shifter",
            condition="one stuck per circuit against none, sigma = 0.02pi",
            system="QGAN, one defect",
            control="QGAN, intact",
            metric="final root fidelity, intended phases",
            status="COMMENSURABLE",
            paper={"one defect": paper.SINGLE_DEFECT, "none": paper.NO_DEFECT},
        )
    )
    noise_free_2500 = _final(runs, "noise_free_reference", "noise_free_2500_rounds")
    for counts, paper_value in zip(paper.SHOT_COUNTS, paper.SHOT_FIDELITY, strict=True):
        rows.append(
            _compare(
                _final(runs, "shot_noise_sweep", f"counts_{counts}"),
                noise_free_2500,
                claim="shot-noise-degradation",
                condition=f"{counts} counts",
                system="QGAN",
                control="Noise-free QGAN",
                metric="final root fidelity, intended phases",
                status="COMMENSURABLE",
                paper=paper_value,
            )
        )
    rows.append(
        _compare(
            _final(runs, "single_runs", "fixed_discriminator"),
            _final(runs, "phase_noise_sweep", "sigma_0.02pi"),
            claim="role of the discriminator (not a paper claim)",
            condition="sigma = 0.02pi, intact, 400 rounds",
            system="Fixed discriminator",
            control="QGAN",
            metric="final root fidelity, intended phases",
            status="COMMENSURABLE",
        )
    )
    return rows


def _by_rounds(item: tuple[str, list[float]]) -> int:
    length = item[0].split()[0]
    return int(length) if length.isdigit() else 0


def systems(runs) -> list[dict]:
    rows = []
    for name, spec in SYSTEMS.items():
        used = [
            (key, value)
            for key, value in runs.items()
            if any(
                key[0] == sweep and run in (None, key[1]) for sweep, run in spec["runs"]
            )
        ]
        expected = sum(
            len(EXPECTED_RUNS[sweep]) if run is None else 1
            for sweep, run in spec["runs"]
        )
        by_length: dict[str, list[float]] = {}
        for _, (metrics, _) in used:
            length = (
                "optimization"  # the ceiling has no rounds
                if metrics["kind"] == "oracle_ceiling"
                else f"{metrics['settings']['rounds']} rounds"
            )
            by_length.setdefault(length, []).append(metrics["seconds"])
        rows.append(
            {
                "system": name,
                **{k: v for k, v in spec.items() if k != "runs"},
                "successful_runs": len(used),
                "expected_runs": expected,
                "targets_per_run": len(SEEDS),
                "wall_clock_minutes": {
                    length: [min(s) / 60, max(s) / 60]
                    for length, s in sorted(by_length.items(), key=_by_rounds)
                },
                "concurrency": CONCURRENCY[spec["provenance"]],
            }
        )
    return rows


def _diff(row: dict) -> str:
    low, high = row["difference"]["ci95"]
    return (
        f"{row['difference']['mean']:+.3f} [{low:+.3f}, {high:+.3f}], {row['outcome']}"
    )


def _pm(row: dict, side: str) -> str:
    return f"{row[f'{side}_mean']:.3f} ± {row[f'{side}_std']:.3f}"


def _find(rows, claim, condition, system, control) -> dict | None:
    for row in rows:
        if (row["claim"], row["condition"], row["system"], row["control"]) == (
            claim,
            condition,
            system,
            control,
        ):
            return row
    return None


def markdown(result: dict) -> str:
    rows = result["comparisons"]
    out = [
        "Phase-05 comparison: the QGAN against its matched baselines. Simulation, "
        "100 targets per run (seeds 0-99), paired by target. Decision rule, "
        "set before these comparisons: tied when the mean paired difference is at "
        "most 0.02; clearly higher or lower when it exceeds 0.02 and its 95 % "
        "interval excludes 0; otherwise no clear separation. Paper values are "
        "read from its figures.",
        "",
        "### Systems",
        "",
        "| System | Role | Provenance | Runs (successful / expected) | Resources | "
        "Wall-clock per run (min) | Status |",
        "|---|---|---|---:|---|---|---|",
    ]
    for s in result["systems"]:
        clock = "; ".join(
            f"{length}: {low:.1f}–{high:.1f}"
            for length, (low, high) in s["wall_clock_minutes"].items()
        )
        out.append(
            f"| {s['system']} | {s['role']} | {s['provenance']} | "
            f"{s['successful_runs']} / {s['expected_runs']} | {s['resources']} | "
            f"{clock} ({s['concurrency']}) | {s['status']} |"
        )
    out += [
        "| Paper QGAN | method as published | paper-reported (Figs. 2c, 3b, "
        "Suppl. Fig. 9b, Suppl. Sec. 6) | — | 1500 counts; training settings "
        "not stated | — | values read from figures |",
        "",
        "Wall-clock: one run = 100 targets; laptop CPU (Apple M5, 10 cores), "
        "one thread per run. Runs that shared the machine are slower, so times "
        "compare workloads only roughly. MerLin with the phase-03 noise draws "
        "gives the numpy results exactly (`merlin_comparison.md`).",
        "",
        "### Phase noise (Fig. 2c): 1500 counts, 400 rounds",
        "",
        "| σ | QGAN | Paper | QGAN − noise-free [95 % CI] | "
        "QGAN − ceiling, noise-averaged [95 % CI] | "
        "MerLin noise − QGAN [95 % CI] |",
        "|---|---:|---:|---|---|---|",
    ]
    claim = "phase-noise-robustness"
    for sigma in SIGMAS:
        condition = f"sigma = {sigma}pi"
        noise_free = _find(rows, claim, condition, "QGAN", "Noise-free QGAN")
        ceiling = _find(rows, claim, condition, "QGAN", "Oracle ceiling")
        merlin = _find(
            rows, claim, condition, "MerLin adaptation, MerLin noise", "QGAN"
        )
        out.append(
            f"| {sigma}π | {_pm(noise_free, 'system')} | {noise_free['paper']:.2f} | "
            f"{_diff(noise_free)} | {_diff(ceiling)} | {_diff(merlin)} |"
        )
    out += [
        "",
        f"Noise-free QGAN, 400 rounds: {_pm(noise_free, 'control')}. Noise-averaged "
        "means, QGAN / ceiling: "
        + "; ".join(
            f"{sigma}π {row['system_mean']:.3f} / {row['control_mean']:.3f}"
            for sigma in SIGMAS
            for row in [
                _find(rows, claim, f"sigma = {sigma}pi", "QGAN", "Oracle ceiling")
            ]
        )
        + ".",
        "",
        "### Stuck phase shifters (Fig. 3b): σ = 0.02π, 1500 counts, 2000 rounds",
        "",
        "| Stuck / 30 | QGAN | Paper | Ceiling | QGAN − ceiling [95 % CI] | "
        "MerLin noise − ceiling [95 % CI] |",
        "|---:|---:|---:|---:|---|---|",
    ]
    claim = "defect-tolerance"
    for stuck in STUCK:
        condition = f"{stuck}/30 stuck"
        qgan = _find(rows, claim, condition, "QGAN", "Oracle ceiling")
        merlin = _find(
            rows, claim, condition, "MerLin adaptation, MerLin noise", "Oracle ceiling"
        )
        out.append(
            f"| {stuck} | {_pm(qgan, 'system')} | {qgan['paper']:.3f} | "
            f"{_pm(qgan, 'control')} | {_diff(qgan)} | "
            f"{_diff(merlin) if merlin else 'not run'} |"
        )
    out += [
        "",
        "### Shot noise (Suppl. Fig. 9b): σ = 0, 2500 rounds",
        "",
        "| Counts | QGAN | Paper | QGAN − noise-free [95 % CI] |",
        "|---:|---:|---:|---|",
    ]
    claim = "shot-noise-degradation"
    for counts in COUNTS:
        row = _find(rows, claim, f"{counts} counts", "QGAN", "Noise-free QGAN")
        out.append(
            f"| {counts} | {_pm(row, 'system')} | {row['paper']:.3f} | {_diff(row)} |"
        )
    single = next(r for r in rows if r["claim"] == "no-crucial-phase-shifter")
    fixed = next(r for r in rows if r["system"] == "Fixed discriminator")
    out += [
        "",
        f"Noise-free QGAN, 2500 rounds: {_pm(row, 'control')}.",
        "",
        "### One stuck phase shifter (Suppl. Sec. 6) and the discriminator",
        "",
        f"- One stuck phase shifter per circuit against none (σ = 0.02π, 2000 "
        f"rounds): {_pm(single, 'system')} against {_pm(single, 'control')}; "
        f"difference {_diff(single)}. Paper: 0.9833 ± 0.0700 against 0.9914 ± 0.0123.",
        f"- Fixed discriminator against the QGAN (σ = 0.02π, intact, 400 rounds): "
        f"{_pm(fixed, 'system')} against {_pm(fixed, 'control')}; difference "
        f"{_diff(fixed)}.",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, help="The paper's outdir/ folder")
    parser.add_argument("--out", type=Path, required=True, help="Results folder")
    args = parser.parse_args()
    runs = _load(args.outdir)
    result = {
        "rule": {
            "tolerance": TOLERANCE,
            "interval": "95 % percentile bootstrap over targets, 10,000 resamples, seed 0",
        },
        "systems": systems(runs),
        "comparisons": comparisons(runs),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    text = markdown(result)
    (args.out / "baseline_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
