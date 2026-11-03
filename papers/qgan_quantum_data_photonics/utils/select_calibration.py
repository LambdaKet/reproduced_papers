"""Apply the predeclared calibration rule to the calibration sweep.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/select_calibration.py <sweep_dir>

Rule (set on 2026-10-30, before any calibration run): keep the
candidates whose mean final fidelity is at least 0.98; pick the one whose mean
fidelity first reaches 0.9 closest to round 59 (the paper's Fig. 2c value at
sigma = 0.01 pi); candidates within 5 rounds of the closest count as tied; among
tied candidates prefer the smaller steps per turn, then the smaller step size,
then the larger generator stop threshold. Writes ``calibration_selection.json``
in the sweep folder and prints it. Exits 1 if the sweep is partial or no
candidate is eligible (calibration gate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_table import build  # noqa: E402

PAPER_ROUND = 59
MIN_FINAL = 0.98
TIE_ROUNDS = 5


def _setting(args: str) -> dict:
    tokens = args.split()
    values = dict(zip(tokens[::2], tokens[1::2], strict=True))
    return {
        "learning_rate_pi": float(values["--learning-rate-pi"]),
        "steps_per_turn": int(values["--steps-per-turn"]),
        "generator_stop_d": float(values["--generator-stop-d"]),
    }


def main(sweep_dir: Path) -> int:
    table = build(sweep_dir)
    candidates = []
    for row in table["rows"]:
        if row["status"] != "COMPLETED":
            continue
        reached = row["first_reaches_0p9"]
        candidates.append(
            {
                "candidate": row["candidate"],
                **_setting(row["args"]),
                "final_mean": row["final_mean"],
                "first_reaches_0p9": reached,
                "eligible": row["final_mean"] >= MIN_FINAL and reached is not None,
                "distance": None if reached is None else abs(reached - PAPER_ROUND),
            }
        )
    eligible = [c for c in candidates if c["eligible"]]
    result = {
        "rule": (
            f"eligible: mean final fidelity >= {MIN_FINAL}; closest round to {PAPER_ROUND}; "
            f"tie within {TIE_ROUNDS} rounds -> smaller steps_per_turn, smaller "
            "learning_rate_pi, larger generator_stop_d"
        ),
        "coverage": table["coverage"],
        "candidates": candidates,
        "tied": [],
        "selected": None,
    }
    if table["coverage"] == "COMPLETE SWEEP" and eligible:
        best = min(c["distance"] for c in eligible)
        tied = [c for c in eligible if c["distance"] <= best + TIE_ROUNDS]
        tied.sort(
            key=lambda c: (
                c["steps_per_turn"],
                c["learning_rate_pi"],
                -c["generator_stop_d"],
            )
        )
        result["tied"] = [c["candidate"] for c in tied]
        result["selected"] = tied[0]
    (sweep_dir / "calibration_selection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
