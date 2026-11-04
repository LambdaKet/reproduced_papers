from __future__ import annotations

import numpy as np
import pytest
from common import (
    PROJECT_DIR,  # noqa: F401  (puts runtime_lib and the DQuLearn package on sys.path)
)
from DQuLearn.lib.circuits import (
    FIDELITY_FLOOR,
    QuClassiLayout,
    build_fidelity_circuit,
    circuit_from_bytes,
    circuit_to_bytes,
    fidelity_from_probability,
)
from qiskit.quantum_info import Statevector


def _exact_fidelity(layout, parameters, angles):
    circuit = build_fidelity_circuit(layout, parameters, angles, measure=False)
    probability_zero = Statevector(circuit).probabilities([0])[0]
    return 2.0 * probability_zero - 1.0


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def _rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


@pytest.mark.parametrize(
    ("qubits", "layers", "expected"),
    [(5, 1, 4), (5, 2, 6), (5, 3, 8), (7, 1, 6), (7, 2, 10), (7, 3, 14)],
)
def test_parameter_counts_follow_quclassi_code(qubits, layers, expected):
    assert QuClassiLayout(qubits, layers).n_parameters == expected


def test_parameter_order_follows_quclassi_variable_dictionary():
    slots = QuClassiLayout(5, 3).parameter_slots
    assert [(slot.gate, slot.qubits) for slot in slots] == [
        ("ry", (1,)),
        ("rz", (1,)),
        ("ryy", (1, 2)),
        ("rzz", (1, 2)),
        ("cry", (1, 2)),
        ("crz", (1, 2)),
        ("ry", (2,)),
        ("rz", (2,)),
    ]


def test_layout_matches_quclassi_code():
    assert QuClassiLayout(5, 1).learned_qubits == (1, 2)
    assert QuClassiLayout(5, 1).data_qubits == (3, 4)
    assert QuClassiLayout(7, 1).learned_qubits == (1, 2, 3)
    assert QuClassiLayout(7, 1).data_qubits == (4, 5, 6)


@pytest.mark.parametrize("qubits", [4, 1])
def test_invalid_width_is_rejected(qubits):
    with pytest.raises(ValueError):
        QuClassiLayout(qubits, 1)


def test_swap_test_returns_the_fidelity_of_the_two_registers():
    # Single style: each register is a product of RZ·RY on |0>, so the fidelity
    # is a product of one-qubit overlaps computed independently here.
    layout = QuClassiLayout(5, 1)
    rng = np.random.default_rng(3)
    parameters = rng.uniform(0, 2 * np.pi, layout.n_parameters)
    angles = rng.uniform(0, np.pi, layout.n_features)
    zero = np.array([1.0, 0.0])
    expected = 1.0
    for qubit in range(layout.register_size):
        learned = _rz(parameters[2 * qubit + 1]) @ _ry(parameters[2 * qubit]) @ zero
        data = _rz(angles[2 * qubit + 1]) @ _ry(angles[2 * qubit]) @ zero
        expected *= abs(np.vdot(learned, data)) ** 2
    assert _exact_fidelity(layout, parameters, angles) == pytest.approx(
        expected, abs=1e-12
    )


def test_identical_states_give_fidelity_one():
    layout = QuClassiLayout(5, 1)
    angles = np.array([0.3, 1.1, 2.0, 0.4])
    # In the Single style the parameters are the same RY, RZ angles as the encoding.
    assert _exact_fidelity(layout, angles, angles) == pytest.approx(1.0, abs=1e-12)


def test_pi_half_shift_gives_the_exact_derivative_of_a_pauli_rotation():
    layout = QuClassiLayout(5, 2)
    rng = np.random.default_rng(5)
    parameters = rng.uniform(0, 2 * np.pi, layout.n_parameters)
    angles = rng.uniform(0, np.pi, layout.n_features)
    for index in range(layout.n_parameters):  # RY, RZ, RYY, RZZ in the Dual style
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        shift_rule = 0.5 * (
            _exact_fidelity(layout, plus, angles)
            - _exact_fidelity(layout, minus, angles)
        )
        step = 1e-6
        plus[index], minus[index] = parameters[index] + step, parameters[index] - step
        finite_difference = (
            _exact_fidelity(layout, plus, angles)
            - _exact_fidelity(layout, minus, angles)
        ) / (2 * step)
        assert shift_rule == pytest.approx(finite_difference, abs=1e-6)


def test_fidelity_estimate_is_floored_like_quclassi():
    assert fidelity_from_probability(0.75) == pytest.approx(0.5)
    assert fidelity_from_probability(0.5) == FIDELITY_FLOOR
    assert fidelity_from_probability(0.4) == FIDELITY_FLOOR


def test_qpy_round_trip_is_exact():
    layout = QuClassiLayout(7, 3)
    rng = np.random.default_rng(7)
    circuit = build_fidelity_circuit(
        layout,
        rng.uniform(0, 2 * np.pi, layout.n_parameters),
        rng.uniform(0, np.pi, layout.n_features),
    )
    assert circuit_from_bytes(circuit_to_bytes(circuit)) == circuit
