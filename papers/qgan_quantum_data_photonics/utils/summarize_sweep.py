"""Build ``sweep_summary.json`` from the ``metrics.json`` files of a sweep.

Usage::

    python papers/qgan_quantum_data_photonics/utils/summarize_sweep.py <sweep_dir>

One row per completed candidate, read from the run's structured metrics only
(never from logs), in the order of the spec. Ranks nothing: selections such as
the calibration rule are applied by the analysis scripts of phase 03.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

QGAN_KEYS = (
    "final_fidelity",
    "noise_averaged_final_fidelity",
    "convergence_round",
    "mean_fidelity_first_reaches",
    "final_d_measured",
)
CEILING_KEYS = ("best_fidelity", "noise_averaged_best_fidelity")


def main(sweep_dir: Path) -> int:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    rows = []
    for run in status["runs"]:
        row = {
            "candidate": run["run_id"],
            "status": run["status"],
            "args": run["candidate"].get("args", []),
        }
        if run["status"] == "COMPLETED" and run["metrics_path"]:
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            keys = QGAN_KEYS if metrics["kind"] == "qgan" else CEILING_KEYS
            if metrics["kind"] == "shot_noise_tvd":
                keys = ("mean_tvd_by_count_level",)
            row.update({key: metrics[key] for key in keys})
            row["metrics_path"] = run["metrics_path"]
        rows.append(row)
    summary = {
        "sweep_id": status["sweep_id"],
        "completed_runs": sum(r["status"] == "COMPLETED" for r in rows),
        "expected_runs": status["expected_runs"],
        "rows": rows,
    }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
