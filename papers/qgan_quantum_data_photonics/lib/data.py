"""Per-seed inputs of a run: targets, discriminator start, stuck phase shifters.

The paper has no external dataset: its learning target is a "randomly
generated" maximally entangled two-ququart state (main text). Following the
Phase 2 data-generation specification in the LOG, every input is drawn from
``numpy.random.SeedSequence(seed)`` with one independent stream per role, so
the target, the discriminator start and the stuck set of a seed do not depend
on the noise level, the count budget or the other seeds of the run (paired
design across conditions).

Our inferences (the paper does not state them): targets tau = (U x 1)|psi0>
with U Haar-random; discriminator phases and stuck values uniform on
[0, 2 pi); k / 2 stuck phase shifters in each generator circuit.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import unitary_group

from .circuits import (
    CHIP_DEFECT_POSITIONS,
    DISCRIMINATOR_NUM_PHASES,
    GENERATOR_NUM_PHASES,
)

PAPER_NAME = "qgan_quantum_data_photonics"
DATASET_NAME = "random_entangled_targets"

# Child index of each random stream in SeedSequence(seed).spawn(...). Fixed
# order: adding a stream at the end never changes the earlier ones.
STREAMS = {
    "target": 0,
    "discriminator_init": 1,
    "stuck_positions": 2,
    "stuck_values": 3,
    "phase_noise": 4,
    "counts": 5,
    "fixed_discriminator": 6,
    "ceiling_restarts": 7,
    "ceiling_noise": 8,
    "evaluation_noise": 9,
}


@dataclass(frozen=True)
class SeedInputs:
    """Inputs of a block of seeds, stacked along the first axis.

    Attributes
    ----------
    seeds : numpy.ndarray
        Shape ``(S,)``, the seed of each target.
    targets : numpy.ndarray
        Shape ``(S, 16)`` complex: |tau> with index 4 * s + i (signal mode s,
        idler mode i, 0-based).
    discriminator_init : numpy.ndarray
        Shape ``(S, 2, 6)``: initial PS1-PS6 of the signal and idler circuits.
    stuck_mask : numpy.ndarray
        Shape ``(S, 2, 15)`` bool: stuck generator phase shifters.
    stuck_values : numpy.ndarray
        Shape ``(S, 2, 15)``: value of each phase shifter if it is stuck.
    """

    seeds: np.ndarray
    targets: np.ndarray
    discriminator_init: np.ndarray
    stuck_mask: np.ndarray
    stuck_values: np.ndarray

    @property
    def target_amplitudes(self) -> np.ndarray:
        """Targets as 4x4 amplitude matrices, shape ``(S, 4, 4)``."""
        return self.targets.reshape(-1, 4, 4)


def seed_stream(seed: int, role: str) -> np.random.Generator:
    """Return the independent random stream of one seed for one role.

    Parameters
    ----------
    seed : int
        Seed of the target.
    role : str
        Key of ``STREAMS``.

    Returns
    -------
    numpy.random.Generator
    """
    children = np.random.SeedSequence(int(seed)).spawn(len(STREAMS))
    return np.random.default_rng(children[STREAMS[role]])


def draw_target(seed: int) -> np.ndarray:
    """Return the target |tau> = (U x 1)|psi0> of a seed, shape ``(16,)``."""
    unitary = unitary_group.rvs(4, random_state=seed_stream(seed, "target"))
    return (0.5 * unitary).reshape(16)


def draw_stuck_mask(seed: int, defects_cfg: dict) -> np.ndarray:
    """Return the stuck generator phase shifters of a seed, shape ``(2, 15)``.

    Parameters
    ----------
    seed : int
        Seed of the target.
    defects_cfg : dict
        ``mode`` is ``none``, ``random_per_circuit`` (``count_per_circuit``
        positions drawn uniformly without replacement in each circuit),
        ``chip`` (the Fig. 1(f) positions in both circuits) or
        ``single_position`` (``position`` stuck in both circuits).

    Returns
    -------
    numpy.ndarray
        Bool array.
    """
    mode = defects_cfg["mode"]
    mask = np.zeros((2, GENERATOR_NUM_PHASES), dtype=bool)
    if mode == "none":
        return mask
    if mode == "random_per_circuit":
        count = int(defects_cfg["count_per_circuit"])
        if not 0 <= count <= GENERATOR_NUM_PHASES:
            raise ValueError(f"count_per_circuit must be in [0, 15], got {count}")
        rng = seed_stream(seed, "stuck_positions")
        for circuit in range(2):
            mask[circuit, rng.permutation(GENERATOR_NUM_PHASES)[:count]] = True
        return mask
    if mode == "chip":
        mask[:, list(CHIP_DEFECT_POSITIONS)] = True
        return mask
    if mode == "single_position":
        position = int(defects_cfg["position"])
        if not 0 <= position < GENERATOR_NUM_PHASES:
            raise ValueError(f"position must be in [0, 14], got {position}")
        mask[:, position] = True
        return mask
    raise ValueError(f"Unknown defects mode {mode!r}")


def generate_inputs(seeds: list[int], defects_cfg: dict) -> SeedInputs:
    """Draw the inputs of a block of seeds.

    Parameters
    ----------
    seeds : list of int
        Seeds of the block; one target per seed.
    defects_cfg : dict
        See ``draw_stuck_mask``.

    Returns
    -------
    SeedInputs
    """
    targets, init, masks, values = [], [], [], []
    for seed in seeds:
        targets.append(draw_target(seed))
        init.append(
            seed_stream(seed, "discriminator_init").uniform(
                0.0, 2.0 * np.pi, size=(2, DISCRIMINATOR_NUM_PHASES)
            )
        )
        masks.append(draw_stuck_mask(seed, defects_cfg))
        # Values are drawn for all 30 positions so they do not depend on k.
        values.append(
            seed_stream(seed, "stuck_values").uniform(
                0.0, 2.0 * np.pi, size=(2, GENERATOR_NUM_PHASES)
            )
        )
    return SeedInputs(
        seeds=np.asarray(seeds, dtype=np.int64),
        targets=np.stack(targets),
        discriminator_init=np.stack(init),
        stuck_mask=np.stack(masks),
        stuck_values=np.stack(values),
    )


def seed_block(cfg: dict) -> list[int]:
    """Return the seeds of a run: ``seeds.start`` to ``start + count - 1``."""
    start, count = int(cfg["seeds"]["start"]), int(cfg["seeds"]["count"])
    if count < 1:
        raise ValueError(f"seeds.count must be at least 1, got {count}")
    return list(range(start, start + count))


def paper_data_dir(cfg: dict) -> Path:
    """Return ``<data root>/qgan_quantum_data_photonics``.

    The root comes from ``--data-root`` (``cfg["data_root"]``), then the
    ``DATA_DIR`` environment variable, then the repository's ``data/``. Nothing
    is read from it: the inputs are generated. It is recorded so that the
    authors' measured data, if ever obtained, have a known home.
    """
    root = cfg.get("data_root") or os.environ.get("DATA_DIR")
    if root is None:
        root = Path(__file__).resolve().parents[3] / "data"
    return Path(root).expanduser().resolve() / PAPER_NAME


def build_dataloader(cfg: dict, split: str) -> list[tuple[np.ndarray, ...]]:
    """Return the run's inputs as a one-batch loader.

    A run learns one target per seed; there is no train, validation or test
    split of data points, so only ``split="train"`` exists.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with ``seeds``, ``defects`` and
        ``dataset.source``.
    split : str
        Must be ``"train"``.

    Returns
    -------
    list of tuple
        One batch ``(targets (S, 16), discriminator_init (S, 2, 6),
        stuck_mask (S, 2, 15), stuck_values (S, 2, 15), seeds (S,))``.
    """
    if split != "train":
        raise ValueError(f"Only the 'train' split exists, got {split!r}")
    source = cfg["dataset"]["source"]
    if source != "generated":
        raise ValueError(
            f"dataset.source must be 'generated' (no external data exists), got {source!r}"
        )
    inputs = generate_inputs(seed_block(cfg), cfg["defects"])
    return [
        (
            inputs.targets,
            inputs.discriminator_init,
            inputs.stuck_mask,
            inputs.stuck_values,
            inputs.seeds,
        )
    ]


def inputs_identity(inputs: SeedInputs) -> str:
    """Return a SHA-256 of the targets and stuck sets, the run's dataset identity."""
    digest = hashlib.sha256()
    for array in (inputs.seeds, inputs.targets, inputs.stuck_mask, inputs.stuck_values):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
