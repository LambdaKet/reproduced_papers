"""Circuit conventions of the supplementary document and the derivations of the plan."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import torch
from qgan_quantum_data_photonics.lib.ceiling import torch_root_fidelity
from qgan_quantum_data_photonics.lib.circuits import (
    CHIP_DEFECT_POSITIONS,
    DISCRIMINATOR_LAYOUT,
    GENERATOR_LAYOUT,
    beam_splitter,
    circuit_unitaries,
    generated_amplitudes,
    generator_identity_phases,
    mzi_matrices,
    output_distribution,
    root_fidelity,
)

RNG = np.random.default_rng(7)


def _random_phases(batch, size):
    return RNG.uniform(0.0, 2.0 * np.pi, (batch, size))


def test_mzi_closed_form_matches_bs_ps_bs():
    theta = RNG.uniform(0, 2 * np.pi, 20)
    bs = beam_splitter()
    for value, matrix in zip(theta, mzi_matrices(theta), strict=True):
        expected = bs @ np.diag([np.exp(1j * value), 1.0]) @ bs
        assert np.abs(matrix - expected).max() < 1e-12


def test_circuits_are_unitary():
    for layout in (GENERATOR_LAYOUT, DISCRIMINATOR_LAYOUT):
        unitaries = circuit_unitaries(_random_phases(10, len(layout)), layout)
        products = unitaries @ np.conj(np.swapaxes(unitaries, -1, -2))
        assert np.abs(products - np.eye(4)).max() < 1e-12


def test_generator_identity_setting():
    unitary = circuit_unitaries(generator_identity_phases()[None], GENERATOR_LAYOUT)[0]
    assert np.abs(unitary - np.eye(4)).max() < 1e-12


def test_generator_mesh_is_locally_universal():
    """15 phases span U(4) up to a global phase (Jacobian rank 15, 16 with it)."""
    phases = _random_phases(1, 15)[0]
    base = circuit_unitaries(phases[None], GENERATOR_LAYOUT)[0]
    columns = []
    for index in range(15):
        shifted = phases.copy()
        shifted[index] += 1e-6
        derivative = (
            circuit_unitaries(shifted[None], GENERATOR_LAYOUT)[0] - base
        ) / 1e-6
        generator = base.conj().T @ derivative
        columns.append(np.concatenate([generator.real.ravel(), generator.imag.ravel()]))
    jacobian = np.array(columns).T
    global_phase = np.concatenate([np.zeros(16), np.eye(4).ravel()])
    assert np.linalg.matrix_rank(jacobian, 1e-6) == 15
    assert np.linalg.matrix_rank(np.column_stack([jacobian, global_phase]), 1e-6) == 16


def test_state_depends_only_on_us_times_ui_transpose():
    signal, idler = _random_phases(5, 15), _random_phases(5, 15)
    amplitudes = generated_amplitudes(signal, idler)
    us = circuit_unitaries(signal, GENERATOR_LAYOUT)
    ui = circuit_unitaries(idler, GENERATOR_LAYOUT)
    assert np.abs(amplitudes - 0.5 * us @ np.swapaxes(ui, -1, -2)).max() < 1e-12


def test_output_distribution_is_normalized_and_bounded():
    amplitudes = generated_amplitudes(_random_phases(50, 15), _random_phases(50, 15))
    probabilities = output_distribution(
        amplitudes, _random_phases(50, 6), _random_phases(50, 6)
    )
    assert np.abs(probabilities.sum(axis=(1, 2)) - 1.0).max() < 1e-12
    # A rank-1 product projector sees at most 1/4 of a maximally entangled state.
    assert probabilities[:, 1, 1].max() <= 0.25 + 1e-12


def test_chip_defect_set_has_eight_positions():
    assert len(set(CHIP_DEFECT_POSITIONS)) == 8
    assert all(0 <= p < 15 for p in CHIP_DEFECT_POSITIONS)


def test_torch_fidelity_matches_numpy():
    phases = _random_phases(6, 30)
    targets = generated_amplitudes(_random_phases(6, 15), _random_phases(6, 15))
    numpy_value = root_fidelity(
        generated_amplitudes(phases[:, :15], phases[:, 15:]), targets
    )
    torch_value = torch_root_fidelity(
        torch.tensor(phases, dtype=torch.float64),
        torch.tensor(targets, dtype=torch.complex128),
    ).numpy()
    assert np.abs(numpy_value - torch_value).max() < 1e-12
