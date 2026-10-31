"""Print the execution ledger of a sweep (Markdown) from ``sweep_status.json``.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/sweep_ledger.py <sweep_dir>

One row per attempt, with the exact command, the evidence paths relative to the
sweep folder, the code commit of the run and the validator's verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main(sweep_dir: Path) -> int:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    print(
        "| Candidate | Targets | Attempt | Status | Command (repository root) | Run log "
        "| Run status | Config snapshot | Metrics | Code commit | Notes |"
    )
    print("|---|---|---:|---|---|---|---|---|---|---|---|")
    for run in status["runs"]:
        for attempt in run["attempts"]:
            rel = attempt["run_dir"]
            commit, targets = "", ""
            if rel:
                run_status = json.loads(
                    (sweep_dir / rel / "run_status.json").read_text()
                )
                commit = run_status["code"]["commit"][:7] + (
                    " (dirty)" if run_status["code"]["dirty"] else ""
                )
                targets = run_status["dataset"]["subset"].split(";")[0]
            final = attempt["attempt"] == run["attempt"]
            verdict = run["status"] if final else "SUPERSEDED"
            if attempt["command"] is None:  # attempt interrupted before it was recorded
                command = "(interrupted; command as in the next attempt)"
            else:
                command = attempt["command"].split("implementation.py", 1)[-1].strip()
                command = "implementation.py " + command.replace(f"{REPO_ROOT}/", "")
            notes = (
                "validated"
                if "passed" in attempt["validation"]
                else attempt["validation"][:80]
            )
            print(
                f"| {run['run_id']} | {targets} | {attempt['attempt']} | {verdict} "
                f"| `{command}` | `{rel}/run.log` | `{rel}/run_status.json` "
                f"| `{rel}/config_snapshot.json` | `{rel}/metrics.json` | `{commit}` | {notes} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
