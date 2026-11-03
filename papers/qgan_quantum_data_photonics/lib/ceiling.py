"""Oracle ceiling: the best fidelity the (possibly defective) generator can reach.

Not in the paper (our control, planned before the experiments). It
maximizes the true root fidelity |<g|tau>| directly over the free generator
phases, with full knowledge of tau, from random restarts. Stuck phase shifters
keep their fixed values. It bounds what any learner could reach with the same
circuit, so the QGAN's gap to it separates circuit expressivity from learning.

Gradients come from torch autograd on the same circuit layout as
``lib.circuits``; Adam is used because its per-coordinate steps keep the
batched restarts independent.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .circuits import GENERATOR_LAYOUT, GENERATOR_NUM_PHASES
from .data import SeedInputs, seed_stream

logger = logging.getLogger(__name__)


def _torch_unitaries(phases: torch.Tensor) -> torch.Tensor:
    """Generator unitaries for phases ``(B, 15)`` (float64), shape ``(B, 4, 4)``."""
    batch = phases.shape[0]
    unitary = torch.eye(4, dtype=torch.complex128, device=phases.device).repeat(
        batch, 1, 1
    )
    for index, (kind, mode) in enumerate(GENERATOR_LAYOUT):
        theta = phases[:, index]
        if kind == "phase":
            factor = torch.ones(
                batch, 4, 1, dtype=torch.complex128, device=phases.device
            )
            factor = factor.index_copy(
                1,
                torch.tensor([mode], device=phases.device),
                torch.exp(1j * theta.to(torch.complex128)).reshape(batch, 1, 1),
            )
            unitary = factor * unitary
        else:
            half = 0.5 * theta.to(torch.complex128)
            prefactor = 1j * torch.exp(1j * half)
            sin, cos = torch.sin(half), torch.cos(half)
            mzi = torch.stack(
                [
                    torch.stack([prefactor * sin, prefactor * cos], dim=-1),
                    torch.stack([prefactor * cos, -prefactor * sin], dim=-1),
                ],
                dim=-2,
            )
            rows = mzi @ unitary[:, mode : mode + 2, :]
            unitary = torch.cat(
                [unitary[:, :mode, :], rows, unitary[:, mode + 2 :, :]], dim=1
            )
    return unitary


def torch_root_fidelity(phases: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Root fidelity for generator phases ``(B, 30)`` and targets ``(B, 4, 4)``."""
    signal = _torch_unitaries(phases[:, :GENERATOR_NUM_PHASES])
    idler = _torch_unitaries(phases[:, GENERATOR_NUM_PHASES:])
    amplitudes = 0.5 * signal @ idler.transpose(-1, -2)
    return torch.abs(torch.sum(torch.conj(amplitudes) * targets, dim=(-2, -1)))


def oracle_ceiling(
    inputs: SeedInputs,
    restarts: int,
    iterations: int,
    learning_rate: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize the true fidelity over the free generator phases of each seed.

    Parameters
    ----------
    inputs : SeedInputs
        Targets and stuck sets.
    restarts : int
        Random starting points per seed (uniform phases on [0, 2 pi)).
    iterations : int
        Adam iterations; one INFO line per iteration.
    learning_rate : float
        Adam step size in radians.
    device : str
        Torch device; complex128 needs ``cpu``.

    Returns
    -------
    tuple of numpy.ndarray
        Best fidelity per seed ``(S,)`` and its phases ``(S, 2, 15)``.
    """
    s = len(inputs.seeds)
    size = 2 * GENERATOR_NUM_PHASES
    starts = np.stack(
        [
            seed_stream(x, "ceiling_restarts").uniform(
                0.0, 2.0 * np.pi, (restarts, size)
            )
            for x in inputs.seeds.tolist()
        ]
    ).reshape(s * restarts, size)
    stuck = np.repeat(inputs.stuck_mask.reshape(s, size), restarts, axis=0)
    values = np.repeat(inputs.stuck_values.reshape(s, size), restarts, axis=0)
    targets = np.repeat(inputs.target_amplitudes, restarts, axis=0)

    dev = torch.device(device)
    free = torch.tensor(starts, dtype=torch.float64, device=dev, requires_grad=True)
    stuck_t = torch.tensor(stuck, device=dev)
    values_t = torch.tensor(values, dtype=torch.float64, device=dev)
    targets_t = torch.tensor(targets, dtype=torch.complex128, device=dev)
    optimizer = torch.optim.Adam([free], lr=learning_rate)

    logger.info(
        "TRAINING_STARTED | ceiling | seeds=%d | restarts=%d | iterations=%d | lr=%.4g",
        s,
        restarts,
        iterations,
        learning_rate,
    )
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad()
        fidelity = torch_root_fidelity(torch.where(stuck_t, values_t, free), targets_t)
        (-fidelity.sum()).backward()
        optimizer.step()
        best = fidelity.detach().reshape(s, restarts).max(dim=1).values
        logger.info(
            "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | mean_best_fidelity=%.6f "
            "| min_best_fidelity=%.6f",
            iteration,
            iterations,
            best.mean().item(),
            best.min().item(),
        )
    with torch.no_grad():
        phases = torch.where(stuck_t, values_t, free)
        fidelity = torch_root_fidelity(phases, targets_t).reshape(s, restarts)
        best_index = fidelity.argmax(dim=1)
        best_phases = phases.reshape(s, restarts, size)[torch.arange(s), best_index]
    return (
        fidelity.max(dim=1).values.cpu().numpy(),
        best_phases.cpu().numpy().reshape(s, 2, GENERATOR_NUM_PHASES),
    )
