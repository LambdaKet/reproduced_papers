"""Time MerLin's ``shots`` sampler against its exact probabilities on the chip layer.

Usage (from the repository root)::

    OMP_NUM_THREADS=1 python papers/qgan_quantum_data_photonics/utils/time_merlin_sampler.py

One batch of 6200 chip executions (random phases, seed 0), the order of the
executions of a few training rounds. Prints both wall-clock times and any
warning raised by MerLin. Supports the phase-04 choice of drawing the counts
with our Poisson model from MerLin's exact probabilities.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for _path in (REPO_ROOT, PROJECT_DIR.parent):  # runtime_lib, and this paper's package
    sys.path.insert(0, str(_path))

from qgan_quantum_data_photonics.lib.qgan_merlin import build_chip_layer  # noqa: E402

EXECUTIONS = 6200
SHOTS = 1500


def main() -> int:
    phases = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, (EXECUTIONS, 42))
    columns = [torch.from_numpy(phases[:, j : j + 1].copy()) for j in range(42)]
    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        layer = build_chip_layer()
        start = time.perf_counter()
        layer(*columns)
        exact = time.perf_counter() - start
        torch.manual_seed(0)
        start = time.perf_counter()
        layer(*columns, shots=SHOTS)
        sampled = time.perf_counter() - start
    print(
        f"{EXECUTIONS} executions: exact probabilities {exact:.3f} s, "
        f"shots={SHOTS} {sampled:.1f} s"
    )
    for message in sorted({str(w.message).splitlines()[0] for w in caught}):
        print(f"warning: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
