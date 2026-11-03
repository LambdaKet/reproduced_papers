"""Write the full candidate table of every sweep to one Markdown file.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/write_sweep_tables.py \\
        papers/qgan_quantum_data_photonics/outdir \\
        --out papers/qgan_quantum_data_photonics/results/sweep_tables.md

Each table comes from ``sweep_table.build`` (structured artifacts only, with
the coverage checks described there). Every sweep is descriptive except the
calibration, whose selection is read from ``calibration_selection.json`` (written
by ``select_calibration.py``). Stops if a sweep is missing or partial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sweep_table

# In reading order: calibration, paper figures, controls, diagnostics, MerLin.
SWEEPS = [
    "calibration",
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
]
HEADER = """\
# Sweep tables

Every candidate of every sweep, generated from the stored run metrics. Common
rules:

- Metric: root fidelity |⟨g|τ⟩| with the phases the generator intends to set,
  averaged over the last 20 rounds of a run (10 for the chip replays);
  higher is better. The oracle ceiling reports the best fidelity over 20
  restarts and its average over phase-noise draws.
- Aggregation: mean ± standard deviation over the targets of a run (one target
  per seed; no data split, since each run learns one state).
- "Round mean reaches 0.9": first round at which the mean over targets reaches
  0.9. "Convergence round": our rule (first round after which the 20-round
  trailing mean stays within 0.01 of the final value); it ends near the run
  length on long runs, so it is not compared with the paper.
- Selection: none, except the calibration below. Each table states its
  coverage (complete when every planned candidate completed with finite
  metrics from one code state) and the code commit (`True`: the working tree
  also held uncommitted files when the runs started).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, help="The paper's outdir/ folder")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file")
    args = parser.parse_args()
    sections = [HEADER]
    for sweep in SWEEPS:
        sweep_dir = args.outdir / f"sweep_{sweep}"
        table = sweep_table.build(sweep_dir)
        if table["coverage"] != "COMPLETE SWEEP":
            raise ValueError(f"{sweep}: {table['problems']}")
        spec = json.loads((sweep_dir / "sweep_spec.json").read_text())
        sections += [f"## {sweep}", "", spec["description"], ""]
        sections.append(sweep_table.markdown(table))
        if sweep == "calibration":
            selection = json.loads(
                (sweep_dir / "calibration_selection.json").read_text()
            )
            sections += [
                "",
                f"Selection rule: {selection['rule']}. Tied candidates: "
                f"{', '.join(selection['tied'])}. Selected: "
                f"{selection['selected']['candidate']} (mean final fidelity "
                f"{selection['selected']['final_mean']:.4f}, mean reaches 0.9 at "
                f"round {selection['selected']['first_reaches_0p9']}).",
            ]
        sections.append("")
    args.out.write_text("\n".join(sections))


if __name__ == "__main__":
    main()
