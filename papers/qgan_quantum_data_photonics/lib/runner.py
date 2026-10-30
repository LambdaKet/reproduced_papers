from __future__ import annotations

import logging
from pathlib import Path


def train_and_evaluate(cfg, run_dir: Path) -> None:
    """Write the scaffold artifact for the QGAN-on-quantum-data reproduction.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.
    """
    # TODO: replace with the QGAN training and evaluation in the implementation
    # phase; this stub only proves that the shared runtime wiring works.
    logger = logging.getLogger(__name__)
    artifact = run_dir / "done.txt"
    artifact.write_text(
        "Scaffold run for qgan_quantum_data_photonics; no experiment implemented yet.\n",
        encoding="utf-8",
    )
    logger.info("Wrote scaffold artifact: %s", artifact)
