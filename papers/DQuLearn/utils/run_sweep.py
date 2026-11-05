"""Run a planned sweep of DQuLearn runs one after another and keep its ledger.

Usage (from the repository root)::

    python papers/DQuLearn/utils/run_sweep.py papers/DQuLearn/utils/sweep_specs/<name>.json \\
        --validator <experiment-logging skill>/scripts/validate_logging.py

The spec lists candidates (a config plus optional CLI flags) and seeds; every
candidate-seed pair is one planned run, its repetition being the seed's index.
``order: "interleaved"`` runs repetition by repetition, candidates in a seeded
random order within each repetition, so that slow drifts of the machine do not
favour one candidate. Each run executes the shared runtime in its own output
folder and writes its own ``run.log``; only this coordinator writes ``sweep.log``
and ``sweep_status.json``. A failed or invalid run gets one automatic second
attempt; after that it stays ``FAILED`` and the sweep ends ``PARTIAL``. When all
runs end, ``summarize_sweep.py`` builds the summary from ``metrics.json``.

Adapted from ``papers/QuaRK/utils/run_sweep.py`` (branch ``PAPER-QuaRK``).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for _path in (REPO_ROOT, PROJECT_DIR.parent):  # runtime_lib, and the DQuLearn package
    sys.path.insert(0, str(_path))

from DQuLearn.lib.experiment_logging import (  # noqa: E402
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


def _validate(validator: str, run_dir: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        validator,
        "--run-dir",
        str(run_dir),
        "--require-training",
        "--require-evaluation",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _relative(path: Path, sweep_dir: Path) -> str:
    return str(path.relative_to(sweep_dir))


def _overlay_config(entry: dict, sweep_id: str, base: Path) -> Path:
    """Candidate config plus the identifiers the run echoes in its run_status.json."""
    candidate = entry["candidate"]
    config = json.loads((PROJECT_DIR / candidate["config"]).read_text())
    config.update(
        sweep_id=sweep_id,
        run_id=entry["run_id"],
        candidate=candidate,
        repetition=entry["repetition"],
    )
    path = base / "config_overlay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def _planned_runs(spec: dict) -> list[dict]:
    """Planned runs in execution order."""
    seeds = list(spec["seeds"])
    candidates = list(spec["candidates"])
    if spec.get("order", "candidate_major") == "interleaved":
        order_seed = int(spec.get("order_seed", 0))
        pairs = []
        for repetition, seed in enumerate(seeds):
            shuffled = list(candidates)
            random.Random(order_seed + repetition).shuffle(shuffled)
            pairs += [(candidate, seed, repetition) for candidate in shuffled]
    else:
        pairs = [
            (candidate, seed, repetition)
            for candidate in candidates
            for repetition, seed in enumerate(seeds)
        ]
    return [
        {
            "run_id": f"{candidate['name']}__seed{seed}",
            "candidate": candidate,
            "seed": seed,
            "repetition": repetition,
            "status": "PLANNED",
            "attempt": 0,
            "attempts": [],
            "run_dir": None,
            "run_log": None,
            "run_status": None,
            "config_path": None,
            "metrics_path": None,
        }
        for candidate, seed, repetition in pairs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help="Default: papers/DQuLearn/outdir/sweep_<sweep_id>",
    )
    parser.add_argument(
        "--validator", required=True, help="Path to validate_logging.py"
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    sweep_dir = (
        args.sweep_dir or PROJECT_DIR / "outdir" / f"sweep_{spec['sweep_id']}"
    ).resolve()
    if (sweep_dir / "sweep_status.json").exists():
        raise SystemExit(f"{sweep_dir} already holds a sweep; choose another folder")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(sweep_dir / "sweep.log")
    (sweep_dir / "sweep_spec.json").write_text(json.dumps(spec, indent=2))

    runs = _planned_runs(spec)
    status = {
        "schema_version": 1,
        "sweep_id": spec["sweep_id"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "expected_runs": len(runs),
        "runs": runs,
        "summary_path": None,
        "selected_candidates": [],
        "error": None,
    }
    write_json(sweep_dir / "sweep_status.json", status)
    logger.info(
        "SWEEP_STARTED | sweep_id=%s | expected_runs=%d | order=%s",
        spec["sweep_id"],
        len(runs),
        spec.get("order", "candidate_major"),
    )

    for entry in status["runs"]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            base = sweep_dir / entry["run_id"] / f"attempt{attempt}"
            overlay = _overlay_config(entry, spec["sweep_id"], base)
            command = [
                sys.executable,
                "implementation.py",
                "--paper",
                "DQuLearn",
                "--config",
                str(overlay),
                "--seed",
                str(entry["seed"]),
                "--outdir",
                str(base),
                *entry["candidate"].get("args", []),
            ]
            entry.update(status="RUNNING", attempt=attempt)
            write_json(sweep_dir / "sweep_status.json", status)
            # One CANDIDATE_STARTED / terminal event per planned run; retries use their own names.
            started = "CANDIDATE_STARTED" if attempt == 1 else "CANDIDATE_RETRY_STARTED"
            logger.info(
                "%s | run_id=%s | candidate=%s | attempt=%d | command=%s",
                started,
                entry["run_id"],
                entry["candidate"]["name"],
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
                else _validate(args.validator, run_dir)
            )
            entry["attempts"].append(
                {
                    "attempt": attempt,
                    "command": " ".join(command),
                    "exit_code": result.returncode,
                    "run_dir": None
                    if run_dir is None
                    else _relative(run_dir, sweep_dir),
                    "validation": report,
                }
            )
            if result.returncode == 0 and valid:
                rel = _relative(run_dir, sweep_dir)
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

    completed = sum(run["status"] == "COMPLETED" for run in status["runs"])
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
        summary_data = json.loads((sweep_dir / "sweep_summary.json").read_text())
        status["selected_candidates"] = summary_data.get("selected", [])
    else:
        logger.error("SUMMARY_FAILED | %s", summary.stderr[-1000:])
    final = (
        "COMPLETED"
        if completed == len(runs) and summary.returncode == 0
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
        len(runs),
        status["summary_path"],
        status["selected_candidates"],
    )
    return 0 if final == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
