"""MerLin adaptation (not the paper's method): a photonic counterpart of QuClassi's circuit.

QuClassi compares a learned state with an encoded data point by a swap test on
qubits. The photonic counterpart used here keeps the data states and the
fidelity, and changes how they are prepared and read:

- each data qubit of QuClassi becomes one photon in two modes (dual rail). A
  beam splitter, a phase ``a``, a beam splitter and a phase ``b`` prepare
  ``cos(a/2)|0> + e^{ib} sin(a/2)|1>``, the state RY(a) then RZ(b) prepare on
  a qubit, so two data points have the same overlap in both encodings;
- the learned class state is ``M(θ)†|s>``, with ``M(θ)`` a trainable universal
  mesh (rectangular interferometer of Mach-Zehnder units) and ``|s>`` the input
  state ``|1,0,1,0,...>``;
- the fidelity is the probability of detecting ``|s>`` after the encoding then
  the mesh: ``|<s|M(θ) E(x)|s>|^2 = |<M(θ)†s|E(x)s>|^2`` (inversion test), with
  no ancilla and no controlled swap.

With ``k`` photons and no loss, the outcome ``|s>`` is ``k`` clicks on the modes
of ``|s>``, so threshold detectors suffice; the probability is computed in
MerLin's FOCK space, which does not renormalise (the UNBUNCHED space keeps only
collision-free outcomes and renormalises over them, a postselection).

A trainable phase enters the probability with Fourier components up to
frequency ``k`` (one per photon through it), so the two-point ±π/2 shift of
DQuLearn's Algorithm 1 does not give the derivative; ``exact_shift_rule`` gives
the general rule with ``2k`` shifts.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

import numpy as np

from .circuits import FIDELITY_FLOOR

COMPUTATION_SPACE = "FOCK"
DETECTOR_MODEL = "threshold"


@dataclass(frozen=True)
class PhotonicLayout:
    """Photonic counterpart of a QuClassi circuit of ``n_qubits`` qubits.

    Parameters
    ----------
    n_qubits : int
        Width of the QuClassi circuit it stands for: one ancilla and two
        registers of ``n_qubits // 2`` qubits. The data register becomes
        ``n_qubits // 2`` photons in twice as many modes.
    n_layers : int, optional
        Must be 1: one universal mesh replaces QuClassi's layer styles.
        Default value is 1.

    Raises
    ------
    ValueError
        If the width is even or below 3, or the layer count is not 1.
    """

    n_qubits: int
    n_layers: int = 1

    def __post_init__(self) -> None:
        if self.n_qubits < 3 or self.n_qubits % 2 == 0:
            raise ValueError(f"n_qubits must be odd and >= 3, got {self.n_qubits}")
        if self.n_layers != 1:
            raise ValueError(
                "the photonic model has one universal mesh: model.layers must be 1, "
                f"got {self.n_layers}"
            )

    @property
    def n_photons(self) -> int:
        return self.n_qubits // 2

    @property
    def n_modes(self) -> int:
        return 2 * self.n_photons

    @property
    def n_features(self) -> int:
        """Data features per circuit: two per dual-rail photon, as two per qubit."""
        return 2 * self.n_photons

    @property
    def input_state(self) -> tuple[int, ...]:
        """One photon in the first mode of each pair: ``(1, 0, 1, 0, ...)``."""
        return (1, 0) * self.n_photons

    @property
    def n_parameters(self) -> int:
        """Phases of the rectangular mesh: two per Mach-Zehnder unit, m(m − 1)/2 units."""
        return self.n_modes * (self.n_modes - 1)

    @property
    def demand(self) -> int:
        """Resource a worker must have free to hold the circuit: its modes."""
        return self.n_modes

    @property
    def max_frequency(self) -> int:
        """Highest Fourier frequency of the fidelity in one phase: the photon number."""
        return self.n_photons


def build_photonic_circuit(layout: PhotonicLayout):
    """Perceval circuit: dual-rail encoding (inputs ``x*``), then the mesh (``theta_*``).

    Returns
    -------
    perceval.Circuit
        Circuit on ``layout.n_modes`` modes with input parameters ``x0 ...``
        (feature order) and trainable parameters ``theta_000 ...``.
    """
    import perceval as pcvl

    circuit = pcvl.Circuit(layout.n_modes)
    for photon in range(layout.n_photons):
        mode = 2 * photon
        circuit.add(mode, pcvl.BS())
        circuit.add(mode, pcvl.PS(pcvl.P(f"x{2 * photon}")))
        circuit.add(mode, pcvl.BS())
        circuit.add(mode, pcvl.PS(pcvl.P(f"x{2 * photon + 1}")))
    counter = iter(range(layout.n_parameters))

    def mach_zehnder(_index: int):
        return (
            pcvl.BS()
            // (0, pcvl.PS(pcvl.P(f"theta_{next(counter):03d}")))
            // pcvl.BS()
            // (0, pcvl.PS(pcvl.P(f"theta_{next(counter):03d}")))
        )

    mesh = pcvl.GenericInterferometer(
        layout.n_modes, mach_zehnder, shape=pcvl.InterferometerShape.RECTANGLE
    )
    circuit.add(0, mesh, merge=True)
    return circuit


def encode_task(
    layout: PhotonicLayout, parameters: np.ndarray, angles: np.ndarray
) -> bytes:
    """Serialize one photonic circuit as float64: photon count, phases, angles."""
    parameters = np.asarray(parameters, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if parameters.shape != (layout.n_parameters,):
        raise ValueError(
            f"expected {layout.n_parameters} parameters, got shape {parameters.shape}"
        )
    if angles.shape != (layout.n_features,):
        raise ValueError(
            f"expected {layout.n_features} angles, got shape {angles.shape}"
        )
    return np.concatenate([[float(layout.n_photons)], parameters, angles]).tobytes()


def decode_task(data: bytes) -> tuple[PhotonicLayout, np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_task`."""
    values = np.frombuffer(data, dtype=np.float64)
    layout = PhotonicLayout(2 * int(values[0]) + 1)
    expected = 1 + layout.n_parameters + layout.n_features
    if values.size != expected:
        raise ValueError(f"expected {expected} values for {layout}, got {values.size}")
    parameters = values[1 : 1 + layout.n_parameters]
    return layout, parameters, values[1 + layout.n_parameters :]


def exact_shift_rule(max_frequency: int) -> tuple[np.ndarray, np.ndarray]:
    """Shifts and weights of the exact derivative of a trigonometric polynomial.

    For ``f(θ)`` with integer frequencies up to ``R``,
    ``f'(θ) = Σ_μ c_μ f(θ + x_μ)`` with ``x_μ = (2μ − 1)π / (2R)`` and
    ``c_μ = (−1)^(μ−1) / (4R sin²(x_μ / 2))``, μ = 1..2R (Wierichs et al.,
    Quantum 6, 677, 2022). ``R = 1`` gives the usual ±π/2 rule.

    Returns
    -------
    tuple of numpy.ndarray
        Shifts ``x_μ`` and weights ``c_μ``, each of length ``2R``.
    """
    if max_frequency < 1:
        raise ValueError(f"max_frequency must be >= 1, got {max_frequency}")
    mu = np.arange(1, 2 * max_frequency + 1)
    shifts = (2 * mu - 1) * np.pi / (2 * max_frequency)
    weights = (-1.0) ** (mu - 1) / (4 * max_frequency * np.sin(shifts / 2) ** 2)
    return shifts, weights


class PhotonicFidelityRunner:
    """Evaluates serialized photonic circuits with MerLin, one at a time.

    Builds one ``QuantumLayer`` per photon count on first use and reads the
    exact probability of detecting the input state (FOCK space, float64).
    Torch runs on one thread, as each worker has one simulation slot.
    """

    def __init__(self) -> None:
        import torch

        torch.set_num_threads(1)
        self._torch = torch
        self._layers: dict[int, tuple] = {}

    def _layer(self, layout: PhotonicLayout):
        cached = self._layers.get(layout.n_photons)
        if cached is None:
            import merlin as ml

            layer = ml.QuantumLayer(
                circuit=build_photonic_circuit(layout),
                input_state=list(layout.input_state),
                n_photons=layout.n_photons,
                input_parameters=["x"],
                trainable_parameters=["theta"],
                measurement_strategy=ml.MeasurementStrategy.probs(
                    computation_space=ml.ComputationSpace.FOCK
                ),
                dtype=self._torch.float64,
            )
            layer.eval()
            (phases,) = [
                parameter
                for name, parameter in layer.named_parameters()
                if name.startswith("theta")
            ]
            if phases.numel() != layout.n_parameters:
                raise RuntimeError(
                    f"MerLin exposes {phases.numel()} trainable phases, "
                    f"expected {layout.n_parameters}"
                )
            keys = [tuple(int(n) for n in key) for key in layer.output_keys]
            cached = (layer, phases, keys.index(layout.input_state))
            self._layers[layout.n_photons] = cached
        return cached

    def probability(
        self, layout: PhotonicLayout, parameters: np.ndarray, angles: np.ndarray
    ) -> float:
        """Exact probability of detecting the input state."""
        layer, phases, index = self._layer(layout)
        torch = self._torch
        with torch.no_grad():
            phases.copy_(torch.tensor(parameters, dtype=torch.float64))
            probabilities = layer(torch.tensor(angles, dtype=torch.float64)[None])
        return float(probabilities[0, index])

    def run(self, data: bytes) -> float:
        """Fidelity of one serialized circuit, floored at 1e-16 as QuClassi's code."""
        layout, parameters, angles = decode_task(data)
        fidelity = self.probability(layout, parameters, angles)
        return fidelity if fidelity > 0.0 else FIDELITY_FLOOR


def hardware_record(layout: PhotonicLayout) -> dict:
    """Hardware-aware fields of a photonic run, for metrics.json."""
    return {
        "computation_space": COMPUTATION_SPACE,
        "detector_model": f"{DETECTOR_MODEL} (outcome |s>: one click on each occupied "
        "input mode; equal to its FOCK probability with no loss)",
        "photons": layout.n_photons,
        "modes": layout.n_modes,
        "input_state": list(layout.input_state),
        "encoding": "dual rail: per photon BS, PS(2 asin sqrt x1), BS, PS(2 asin sqrt x2)",
        "trainable": f"rectangular Mach-Zehnder mesh, {layout.n_parameters} phases",
        "measurement": "MeasurementStrategy.probs(FOCK); probability of the input state",
        "postselection": "none",
        "simulator": "MerLin CPU simulator, exact probabilities",
        "shots": None,
        "merlinquantum": importlib.metadata.version("merlinquantum"),
        "perceval_quandela": importlib.metadata.version("perceval-quandela"),
        "torch_threads_per_process": 1,
    }
