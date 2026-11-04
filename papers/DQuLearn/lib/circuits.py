"""QuClassi circuits (Stein et al., MLSys 2022), ported to Qiskit 2.x.

DQuLearn adopts the circuit designs of QuClassi's open-source code (DQuLearn
Sec. IV-A). Layout, gates and parameter order follow that code
(``QuClassiExample/main.py``, commit ``8e85e08``):

- qubit 0 is the swap-test ancilla;
- qubits ``1..h`` hold the learned (class) state, ``h = n_qubits // 2``;
- qubits ``h+1..2h`` hold the data point, two features per qubit (RY, then RZ);
- the swap test applies H on the ancilla, controlled-SWAPs between paired
  learned and data qubits, H again, then measures the ancilla.

DQuLearn's 1, 2 and 3 layers (Sec. IV-A, Experiment Settings) are QuClassi's
layer styles ``Single`` (RY, RZ per qubit), ``Dual`` (adds RYY, RZZ on
neighbouring qubits) and ``Controlled-Dual`` (adds CRY, CRZ on neighbours).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from qiskit import QuantumCircuit, qpy

LAYER_STYLES = {1: "Single", 2: "Dual", 3: "Controlled-Dual"}
FIDELITY_FLOOR = 1e-16

# QuClassi's code applies every single-qubit unitary first, then the dual-qubit
# unitaries, then the controlled ones, whatever the parameter order.
_APPLICATION_ORDER = (("ry", "rz"), ("ryy", "rzz"), ("cry", "crz"))


@dataclass(frozen=True)
class GateSlot:
    """One trainable rotation of the learned state.

    Parameters
    ----------
    gate : str
        Qiskit gate name: ``ry``, ``rz``, ``ryy``, ``rzz``, ``cry`` or ``crz``.
    qubits : tuple of int
        Qubits the gate acts on; for ``cry`` and ``crz`` the first is the control.
    """

    gate: str
    qubits: tuple[int, ...]


@dataclass(frozen=True)
class QuClassiLayout:
    """Qubit layout and trainable parameters of one QuClassi circuit.

    Parameters
    ----------
    n_qubits : int
        Circuit width: one ancilla plus two registers of equal size, so odd.
    n_layers : int
        DQuLearn's layer count, 1 to 3 (see ``LAYER_STYLES``).

    Raises
    ------
    ValueError
        If the width is even or below 3, or the layer count is not 1, 2 or 3.
    """

    n_qubits: int
    n_layers: int

    def __post_init__(self) -> None:
        if self.n_qubits < 3 or self.n_qubits % 2 == 0:
            raise ValueError(f"n_qubits must be odd and >= 3, got {self.n_qubits}")
        if self.n_layers not in LAYER_STYLES:
            raise ValueError(f"n_layers must be 1, 2 or 3, got {self.n_layers}")

    @property
    def style(self) -> str:
        """QuClassi layer style matching the layer count."""
        return LAYER_STYLES[self.n_layers]

    @property
    def register_size(self) -> int:
        """Qubits in the learned register, and in the data register."""
        return self.n_qubits // 2

    @property
    def learned_qubits(self) -> tuple[int, ...]:
        return tuple(range(1, self.register_size + 1))

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.register_size + 1, 2 * self.register_size + 1))

    @property
    def n_features(self) -> int:
        """Data features encoded per circuit: two per data qubit."""
        return 2 * self.register_size

    @cached_property
    def parameter_slots(self) -> tuple[GateSlot, ...]:
        """Trainable rotations in the order of QuClassi's variable dictionary.

        For each learned qubit ``i``: RY and RZ on ``i``; then, except for the
        last qubit, RYY and RZZ on ``(i, i+1)`` (2 and 3 layers) and CRY and CRZ
        on ``(i, i+1)`` (3 layers). QuClassi's sequential update visits the
        parameters in this order.
        """
        slots: list[GateSlot] = []
        for position, qubit in enumerate(self.learned_qubits):
            slots += [GateSlot("ry", (qubit,)), GateSlot("rz", (qubit,))]
            if position < self.register_size - 1 and self.n_layers >= 2:
                pair = (qubit, qubit + 1)
                slots += [GateSlot("ryy", pair), GateSlot("rzz", pair)]
                if self.n_layers >= 3:
                    slots += [GateSlot("cry", pair), GateSlot("crz", pair)]
        return tuple(slots)

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_slots)


def build_fidelity_circuit(
    layout: QuClassiLayout,
    parameters: np.ndarray,
    angles: np.ndarray,
    *,
    measure: bool = True,
) -> QuantumCircuit:
    """Swap-test circuit between a learned state and one encoded data point.

    Parameters
    ----------
    layout : QuClassiLayout
        Circuit width and layer count.
    parameters : numpy.ndarray
        Learned-state rotations, shape ``(layout.n_parameters,)``, in the order
        of ``layout.parameter_slots``.
    angles : numpy.ndarray
        Encoding angles ``2·asin(√x)``, shape ``(layout.n_features,)``.
    measure : bool, optional
        Measure the ancilla into the single classical bit. Default value is True.

    Returns
    -------
    qiskit.QuantumCircuit
        Circuit whose ancilla reads 0 with probability ``(1 + F) / 2``, where F
        is the fidelity between the learned state and the data state.

    Raises
    ------
    ValueError
        If ``parameters`` or ``angles`` do not match the layout.
    """
    parameters = np.asarray(parameters, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if parameters.shape != (layout.n_parameters,):
        raise ValueError(
            f"expected {layout.n_parameters} parameters, got shape {parameters.shape}"
        )
    if angles.shape != (layout.n_features,):
        raise ValueError(
            f"expected {layout.n_features} angles, got shape {angles.shape}"
        )

    circuit = QuantumCircuit(layout.n_qubits, 1)
    circuit.h(0)
    for gate_group in _APPLICATION_ORDER:
        for index, slot in enumerate(layout.parameter_slots):
            if slot.gate in gate_group:
                getattr(circuit, slot.gate)(float(parameters[index]), *slot.qubits)
    for position, qubit in enumerate(layout.data_qubits):
        circuit.ry(float(angles[2 * position]), qubit)
        circuit.rz(float(angles[2 * position + 1]), qubit)
    for learned, data in zip(layout.learned_qubits, layout.data_qubits, strict=True):
        circuit.cswap(0, learned, data)
    circuit.h(0)
    if measure:
        circuit.measure(0, 0)
    return circuit


def fidelity_from_probability(probability_zero: float) -> float:
    """QuClassi's fidelity estimate from the ancilla's probability of reading 0.

    The swap test gives ``P(0) = (1 + F) / 2``, so ``F = 2·P(0) − 1``. QuClassi's
    code floors non-positive estimates at ``1e-16`` so that ``−log F`` stays finite.

    Parameters
    ----------
    probability_zero : float
        Measured or exact probability of the ancilla reading 0.

    Returns
    -------
    float
        Fidelity estimate in ``[1e-16, 1]``.
    """
    fidelity = 2.0 * float(probability_zero) - 1.0
    return fidelity if fidelity > 0.0 else FIDELITY_FLOOR


def circuit_to_bytes(circuit: QuantumCircuit) -> bytes:
    """Serialize a circuit with QPY, the form in which the manager sends it."""
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    return buffer.getvalue()


def circuit_from_bytes(data: bytes) -> QuantumCircuit:
    """Deserialize a QPY circuit received from the manager."""
    return qpy.load(io.BytesIO(data))[0]
