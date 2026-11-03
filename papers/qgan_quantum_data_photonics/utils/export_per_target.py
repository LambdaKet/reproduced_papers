"""Export the per-target results the notebook needs into one tracked JSON file.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/export_per_target.py \\
        papers/qgan_quantum_data_photonics/outdir \\
        --out papers/qgan_quantum_data_photonics/results/per_target_results.json

The raw runs under ``outdir/`` are not published; this file keeps, for every
run of the listed sweeps, its seeds, settings, wall-clock and per-target values
(final root fidelity; for the QGAN, the noise-averaged fidelity at the last
round; for the ceiling, the best fidelity and its noise averages) and, for the
Fig. 2c and noise-free runs, the mean learning curve, so the
notebook can recompute every paired comparison. Reads only the files listed by
COMPLETED ``sweep_status.json``; stops on a missing or partial sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SWEEPS = (
    "phase_noise_sweep",
    "phase_noise_variants",
    "phase_noise_diagnosis",
    "defect_rate_sweep_low",
    "defect_rate_sweep_mid",
    "defect_rate_sweep_high",
    "shot_noise_sweep",
    "single_runs",
    "single_defect_positions",
    "chip_emulations",
    "oracle_ceiling",
    "noise_free_reference",
    "merlin_equivalence",
    "merlin_native_phase_noise",
    "merlin_native_defects",
)
DIGITS = 6
# Mean learning curves kept only where the notebook plots them (file size).
CURVE_SWEEPS = ("sweep_phase_noise_sweep", "sweep_noise_free_reference")


def _rounded(values) -> list[float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("non-finite per-target value")
    return [round(float(v), DIGITS) for v in array]


def _run_record(sweep_dir: Path, run: dict) -> dict:
    metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
    record = {
        "seeds": metrics["seeds"],
        "stuck_per_seed": metrics["stuck_per_seed"],
        "seconds": metrics["seconds"],
        "kind": metrics["kind"],
    }
    if metrics["kind"] == "oracle_ceiling":
        record["best_fidelity"] = _rounded(metrics["best_fidelity_per_seed"])
        record["noise_averaged_best_fidelity"] = {
            sigma: _rounded(values)
            for sigma, values in metrics[
                "noise_averaged_best_fidelity_per_seed"
            ].items()
        }
        return record
    arrays = np.load(sweep_dir / run["run_dir"] / metrics["artifacts"]["trajectories"])
    if int(arrays["noise_averaged_rounds"][-1]) != metrics["settings"]["rounds"]:
        raise ValueError(f"{run['run_id']}: no noise average at the last round")
    record["settings"] = metrics["settings"]
    record["final_fidelity"] = _rounded(metrics["final_fidelity_per_seed"])
    record["noise_averaged_final_fidelity"] = _rounded(arrays["noise_averaged"][-1])
    if sweep_dir.name in CURVE_SWEEPS:
        record["mean_fidelity_by_round"] = _rounded(metrics["mean_fidelity_by_round"])
    if "hardware" in metrics:
        record["hardware"] = metrics["hardware"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, help="The paper's outdir/ folder")
    parser.add_argument("--out", type=Path, required=True, help="JSON file")
    args = parser.parse_args()
    export = {}
    for sweep in SWEEPS:
        sweep_dir = args.outdir / f"sweep_{sweep}"
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        expected = json.loads((sweep_dir / "sweep_spec.json").read_text())
        planned = sorted(c["name"] for c in expected["candidates"])
        if status["status"] != "COMPLETED" or planned != sorted(
            r["run_id"] for r in status["runs"]
        ):
            raise ValueError(f"{sweep} is not a complete sweep")
        export[sweep] = {
            run["run_id"]: _run_record(sweep_dir, run) for run in status["runs"]
        }
    args.out.write_text(json.dumps(export, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
