"""Rebuild every file in results/ from outdir/, then check the Fig. 3b values.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/regenerate_results.py

Runs, in order, the scripts below with ``outdir/`` as input and ``results/`` as
output; each one can also be run on its own. Then checks the Fig. 3b values of
``paper_values.py`` against the paper's vector figure, read from the arXiv
source archive under the shared data root. Stops at the first failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parent
PAPER_DIR = UTILS.parent
REPO_ROOT = PAPER_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from runtime_lib import resolve_data_root  # noqa: E402

OUTDIR = PAPER_DIR / "outdir"
RESULTS = PAPER_DIR / "results"
# (script, output given to --out)
STEPS = [
    ("compare_with_paper.py", RESULTS),  # comparison_with_paper.*
    ("plot_results.py", RESULTS),  # phase_noise, defect_rate, shot_noise .png
    ("compare_merlin.py", RESULTS),  # merlin_comparison.*
    ("compare_baselines.py", RESULTS),  # baseline_comparison.*
    ("write_sweep_tables.py", RESULTS / "sweep_tables.md"),
    ("export_per_target.py", RESULTS / "per_target_results.json"),
]
FIG3 = "qgan_quantum_data_photonics/sources/arxiv_2310.00585v1_src/fig3.pdf"


def run(arguments: list[str]) -> None:
    """Run one utility with the current interpreter; stop on failure."""
    shown = [
        str(Path(a).relative_to(REPO_ROOT)) if a.startswith(str(REPO_ROOT)) else a
        for a in arguments
    ]
    print("$ python " + " ".join(shown), flush=True)
    subprocess.run([sys.executable, *arguments], check=True)


def main() -> None:
    for script, out in STEPS:
        run([str(UTILS / script), str(OUTDIR), "--out", str(out)])
    fig3 = resolve_data_root() / FIG3
    if not fig3.exists():
        raise FileNotFoundError(
            f"{fig3} not found: download and unpack the arXiv source "
            "(https://arxiv.org/e-print/2310.00585v1) to check the Fig. 3b values"
        )
    run([str(UTILS / "read_figure_3b.py"), str(fig3), "--check"])


if __name__ == "__main__":
    main()
