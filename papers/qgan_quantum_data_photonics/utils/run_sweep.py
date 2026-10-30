"""Run a planned sweep of runs one after another and keep its ledger.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/run_sweep.py \\
        papers/qgan_quantum_data_photonics/utils/sweep_specs/<name>.json \\
        --sweep-dir papers/qgan_quantum_data_photonics/outdir/sweep_<name> \\
        --validator <path to experiment-logging/scripts/validate_logging.py>

The spec lists candidates (a config plus CLI flags). Each candidate is one run
of the shared runtime over its whole block of target seeds, in its own output
folder with its own ``run.log``; only this coordinator writes ``sweep.log`` and
``sweep_status.json``. A failed or invalid run gets one second attempt; after
that it stays ``FAILED`` and the sweep ends ``PARTIAL``. When all runs end,
``summarize_sweep.py`` builds the summary from the ``metrics.json`` files.
Pattern adapted from ``papers/QuaRK/utils/run_sweep.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for _path in (REPO_ROOT, PROJECT_DIR.parent):  # runtime_lib, and this paper's package
    sys.path.insert(0, str(_path))

from qgan_quantum_data_photonics.lib.experiment_logging import (  # noqa: E402
    configure_logging,
    git_state,
    sweep_progress,
    utc_now,
    write_json,
)

MAX_ATTEMPTS = 2
logger = logging.getLogger("sweep")


def _run_dir_of(base: Path) -> Path | None:
    runs = sorted(base.glob("run_*"))
    return runs[-1] if runs else None


def _validate(
    validator: str, run_dir: Path, require_training: bool
) -> tuple[bool, str]:
    command = [
        sys.executable,
        validator,
        "--run-dir",
        str(run_dir),
        "--require-evaluation",
    ]
    if require_training:
        command.append("--require-training")
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _overlay_config(candidate: dict, sweep_id: str, run_id: str, base: Path) -> Path:
    """Candidate config plus the identifiers the run must echo in its run_status.json."""
    config = json.loads((PROJECT_DIR / candidate["config"]).read_text())
    config.update(sweep_id=sweep_id, run_id=run_id, candidate=candidate, repetition=0)
    path = base / "config_overlay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--validator", required=True, help="Path to validate_logging.py"
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    sweep_dir = args.sweep_dir.resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")
    (sweep_dir / "sweep_spec.json").write_text(json.dumps(spec, indent=2))
    require_training = bool(spec.get("require_training", True))
    seed = int(spec.get("seed", 0))

    status = {
        "schema_version": 1,
        "sweep_id": spec["sweep_id"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "expected_runs": len(spec["candidates"]),
        "runs": [
            {
                "run_id": candidate["name"],
                "candidate": candidate,
                "seed": seed,
                "repetition": 0,
                "status": "PLANNED",
                "attempt": 0,
                "attempts": [],
                "run_dir": None,
                "run_log": None,
                "run_status": None,
                "config_path": None,
                "metrics_path": None,
            }
            for candidate in spec["candidates"]
        ],
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d",
        spec["sweep_id"],
        status["expected_runs"],
    )

    for entry in status["runs"]:
        candidate = entry["candidate"]
        for attempt in range(1, MAX_ATTEMPTS + 1):
            base = sweep_dir / entry["run_id"] / f"attempt{attempt}"
            overlay = _overlay_config(
                candidate, spec["sweep_id"], entry["run_id"], base
            )
            command = [
                sys.executable,
                "implementation.py",
                "--paper",
                "qgan_quantum_data_photonics",
                "--config",
                str(overlay),
                "--seed",
                str(seed),
                "--outdir",
                str(base),
                *candidate.get("args", []),
            ]
            entry.update(status="RUNNING", attempt=attempt)
            write_json(sweep_dir / "sweep_status.json", status)
            started = "CANDIDATE_STARTED" if attempt == 1 else "CANDIDATE_RETRY_STARTED"
            logger.info(
                "%s | run_id=%s | attempt=%d | command=%s",
                started,
                entry["run_id"],
                attempt,
                " ".join(command),
            )
            logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))
            result = subprocess.run(
                command, cwd=REPO_ROOT, capture_output=True, text=True
            )
            run_dir = _run_dir_of(base)
            valid, report = (
                (False, "no run directory")
                if run_dir is None
                else _validate(args.validator, run_dir, require_training)
            )
            entry["attempts"].append(
                {
                    "attempt": attempt,
                    "command": " ".join(command),
                    "exit_code": result.returncode,
                    "run_dir": None
                    if run_dir is None
                    else str(run_dir.relative_to(sweep_dir)),
                    "validation": report,
                }
            )
            if result.returncode == 0 and valid:
                rel = str(run_dir.relative_to(sweep_dir))
                entry.update(
                    status="COMPLETED",
                    run_dir=rel,
                    run_log=f"{rel}/run.log",
                    run_status=f"{rel}/run_status.json",
                    config_path=f"{rel}/config_snapshot.json",
                    metrics_path=f"{rel}/metrics.json",
                )
                write_json(sweep_dir / "sweep_status.json", status)
                logger.info(
                    "CANDIDATE_COMPLETED | run_id=%s | metrics=%s",
                    entry["run_id"],
                    entry["metrics_path"],
                )
                break
            final_attempt = attempt == MAX_ATTEMPTS
            entry["status"] = "FAILED" if final_attempt else "RUNNING"
            write_json(sweep_dir / "sweep_status.json", status)
            failed = "CANDIDATE_FAILED" if final_attempt else "CANDIDATE_ATTEMPT_FAILED"
            logger.error(
                "%s | run_id=%s | attempt=%d | exit=%d | validation=%s | stderr_tail=%s",
                failed,
                entry["run_id"],
                attempt,
                result.returncode,
                report,
                result.stderr[-500:],
            )
        logger.info("SWEEP_PROGRESS | %s", sweep_progress(status))

    completed = sum(r["status"] == "COMPLETED" for r in status["runs"])
    summary = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "utils" / "summarize_sweep.py"),
            str(sweep_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if summary.returncode == 0:
        status["summary_path"] = "sweep_summary.json"
    else:
        logger.error("SUMMARY_FAILED | %s", summary.stderr[-1000:])
    final = (
        "COMPLETED"
        if completed == len(status["runs"]) and summary.returncode == 0
        else ("PARTIAL" if completed else "FAILED")
    )
    status.update(status=final, completed_at=utc_now())
    if final == "FAILED":
        status["error"] = {
            "type": "NoCompletedRuns",
            "message": "no planned run completed",
        }
    write_json(sweep_dir / "sweep_status.json", status)
    event = {
        "COMPLETED": "SWEEP_COMPLETED",
        "PARTIAL": "SWEEP_PARTIAL",
        "FAILED": "SWEEP_FAILED",
    }[final]
    logger.info(
        "%s | completed=%d/%d | summary=%s | selected=%s",
        event,
        completed,
        len(status["runs"]),
        status["summary_path"],
        status["selected_candidates"],
    )
    return 0 if final == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
