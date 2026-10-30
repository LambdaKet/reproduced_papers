"""Chip circuits of the paper as batched single-photon 4x4 unitaries (numpy).

Conventions follow the supplementary document (Suppl. Eqs. 2-4): the multimode
interferometer is BS = [[1, i], [i, 1]] / sqrt(2), a phase shifter is
PS(theta) = diag(exp(i theta), 1) on the upper arm, and MZI(theta) =
BS @ PS(theta) @ BS. Modes are 0-based here: paper mode k is index k - 1.

Each photon goes through its own 4-mode circuit, so a circuit is a 4x4 unitary
acting on the photon's path amplitudes. Operations are applied left to right
in the order listed in the layouts below.
"""

from __future__ import annotations

import numpy as np

NUM_MODES = 4

# Generator mesh of Suppl. Fig. 2(d), one per photon: 6 MZIs and 15 phase
# shifters. ("phase", mode) is an external phase shifter on that mode;
# ("mzi", upper) is an MZI on modes (upper, upper + 1) whose internal phase
# shifter sits on the upper arm. The list index is the phase-shifter index.
GENERATOR_LAYOUT: tuple[tuple[str, int], ...] = (
    ("phase", 1),  # step 1: input phases on paper modes 2, 3, 4
    ("phase", 2),
    ("phase", 3),
    ("mzi", 1),  # step 2: MZI(2,3)
    ("mzi", 0),  # step 3: MZI(1,2) and MZI(3,4)
    ("mzi", 2),
    ("phase", 1),  # step 4: phase on paper mode 2
    ("mzi", 1),  # step 5: MZI(2,3)
    ("phase", 0),  # step 6: phases on paper modes 1 and 3
    ("phase", 2),
    ("mzi", 0),  # step 7: MZI(1,2) and MZI(3,4)
    ("mzi", 2),
    ("phase", 0),  # step 8: output phases on paper modes 1, 2, 3
    ("phase", 1),
    ("phase", 2),
)
GENERATOR_NUM_PHASES = len(GENERATOR_LAYOUT)

# Projection circuit of Suppl. Fig. 3, one per photon: PS1-PS3 on paper modes
# 1-3, MZI(1,2) with PS4, MZI(3,4) with PS5, then MZI(2,3) with PS6. The
# photon is detected at output port 2 (index 1).
DISCRIMINATOR_LAYOUT: tuple[tuple[str, int], ...] = (
    ("phase", 0),
    ("phase", 1),
    ("phase", 2),
    ("mzi", 0),
    ("mzi", 2),
    ("mzi", 1),
)
DISCRIMINATOR_NUM_PHASES = len(DISCRIMINATOR_LAYOUT)
DETECTION_PORT = 1

# Phase shifters stuck in the chip run of Fig. 3a, read from Fig. 1(f): the
# three input phases (step 1), the internal phase of the second MZI(2,3)
# (step 5), the two phases of step 6 and the internal phases of the last
# MZI(1,2) and MZI(3,4) (step 7). The figure shows one circuit; we use the
# same positions in both (our inference).
CHIP_DEFECT_POSITIONS: tuple[int, ...] = (0, 1, 2, 7, 8, 9, 10, 11)


def beam_splitter() -> np.ndarray:
    """Return the 2x2 multimode-interferometer matrix of Suppl. Eq. 2."""
    return np.array([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)


def mzi_matrices(theta: np.ndarray) -> np.ndarray:
    """Return MZI(theta) = BS @ PS(theta) @ BS for a batch of phases.

    Parameters
    ----------
    theta : numpy.ndarray
        Internal phases, any shape ``S``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``S + (2, 2)``, the closed form of Suppl. Eq. 4.
    """
    half = 0.5 * np.asarray(theta, dtype=float)
    prefactor = 1.0j * np.exp(1.0j * half)
    sin, cos = np.sin(half), np.cos(half)
    out = np.empty(half.shape + (2, 2), dtype=complex)
    out[..., 0, 0] = prefactor * sin
    out[..., 0, 1] = prefactor * cos
    out[..., 1, 0] = prefactor * cos
    out[..., 1, 1] = -prefactor * sin
    return out


def circuit_unitaries(
    phases: np.ndarray, layout: tuple[tuple[str, int], ...]
) -> np.ndarray:
    """Build the 4x4 unitaries of a circuit for a batch of phase vectors.

    Parameters
    ----------
    phases : numpy.ndarray
        Real array of shape ``(B, len(layout))``, one phase per layout entry.
    layout : tuple of (str, int)
        Circuit layout, ``GENERATOR_LAYOUT`` or ``DISCRIMINATOR_LAYOUT``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(B, 4, 4)``: output amplitudes = U @ input.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 2 or phases.shape[1] != len(layout):
        raise ValueError(
            f"Expected phases of shape (B, {len(layout)}), got {phases.shape}"
        )
    batch = phases.shape[0]
    unitary = np.broadcast_to(np.eye(NUM_MODES, dtype=complex), (batch, 4, 4)).copy()
    for index, (kind, mode) in enumerate(layout):
        theta = phases[:, index]
        if kind == "phase":
            unitary[:, mode, :] *= np.exp(1.0j * theta)[:, None]
        elif kind == "mzi":
            rows = unitary[:, mode : mode + 2, :]
            unitary[:, mode : mode + 2, :] = mzi_matrices(theta) @ rows
        else:
            raise ValueError(f"Unknown layout operation {kind!r}")
    return unitary


def generator_identity_phases() -> np.ndarray:
    """Return generator phases that implement the identity.

    With every internal MZI phase at pi, MZI(pi) = diag(-1, 1); each of paper
    modes 1-3 is the upper mode of exactly two MZIs and mode 4 of none, so the
    signs cancel and all external phases can stay at 0.

    Returns
    -------
    numpy.ndarray
        Shape ``(15,)``.
    """
    phases = np.zeros(GENERATOR_NUM_PHASES)
    for index, (kind, _) in enumerate(GENERATOR_LAYOUT):
        if kind == "mzi":
            phases[index] = np.pi
    return phases


def generated_amplitudes(
    signal_phases: np.ndarray, idler_phases: np.ndarray
) -> np.ndarray:
    """Return the two-photon amplitude matrix of the generated state.

    The input |psi0> = 1/2 sum_k |k>_s |k>_i has amplitude matrix I / 2; after
    the generator, A = U_s (I / 2) U_i^T with A[s, i] the amplitude of signal
    mode s and idler mode i (main text Eq. 3).

    Parameters
    ----------
    signal_phases, idler_phases : numpy.ndarray
        Shape ``(B, 15)`` each.

    Returns
    -------
    numpy.ndarray
        Shape ``(B, 4, 4)``.
    """
    signal = circuit_unitaries(signal_phases, GENERATOR_LAYOUT)
    idler = circuit_unitaries(idler_phases, GENERATOR_LAYOUT)
    return 0.5 * signal @ np.swapaxes(idler, -1, -2)


def output_distribution(
    amplitudes: np.ndarray, signal_phases: np.ndarray, idler_phases: np.ndarray
) -> np.ndarray:
    """Return the coincidence probabilities after the projection circuits.

    Parameters
    ----------
    amplitudes : numpy.ndarray
        Two-photon amplitude matrices, shape ``(B, 4, 4)``.
    signal_phases, idler_phases : numpy.ndarray
        Discriminator phases of each photon, shape ``(B, 6)`` each.

    Returns
    -------
    numpy.ndarray
        Shape ``(B, 4, 4)``: probability that the signal exits port a and the
        idler port b; entry ``[:, 1, 1]`` is tr[M rho] (Suppl. Eq. 11).
    """
    signal = circuit_unitaries(signal_phases, DISCRIMINATOR_LAYOUT)
    idler = circuit_unitaries(idler_phases, DISCRIMINATOR_LAYOUT)
    out = signal @ amplitudes @ np.swapaxes(idler, -1, -2)
    return np.abs(out) ** 2


def root_fidelity(amplitudes: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return |<g|tau>|, the root fidelity of two pure states (Suppl. Eq. 12).

    Parameters
    ----------
    amplitudes : numpy.ndarray
        Generated amplitude matrices, shape ``(..., 4, 4)``.
    target : numpy.ndarray
        Target amplitude matrices, broadcastable to ``amplitudes``.

    Returns
    -------
    numpy.ndarray
        Shape ``(...)``.
    """
    return np.abs(np.sum(np.conj(amplitudes) * target, axis=(-2, -1)))
