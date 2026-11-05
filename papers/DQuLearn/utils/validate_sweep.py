"""Validate a sweep's coordinator artifacts and every run it recorded, in one call.

Usage (from the repository root)::

    python papers/DQuLearn/utils/validate_sweep.py <sweep_dir> \\
        --validator <experiment-logging skill>/scripts/validate_logging.py

Reads ``sweep_status.json``, passes each recorded run directory with
``--run-dir`` and the sweep folder with ``--sweep-dir`` to the experiment-logging
validator, prints its report and returns its exit status.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument(
        "--validator", required=True, help="Path to validate_logging.py"
    )
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    runs = json.loads((sweep_dir / "sweep_status.json").read_text())["runs"]
    run_args = [
        item
        for run in runs
        if run.get("run_dir")
        for item in ("--run-dir", str(sweep_dir / run["run_dir"]))
    ]
    command = [
        sys.executable,
        args.validator,
        *run_args,
        "--sweep-dir",
        str(sweep_dir),
        "--require-training",
        "--require-evaluation",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(result.stdout + result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
