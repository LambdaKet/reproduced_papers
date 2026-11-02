"""MerLin adaptation: the paper's chip as a ``merlin.QuantumLayer``.

Phase-04 adaptation, not part of the reproduction itself. The chip of Suppl.
Fig. 1 is one 8-mode circuit (signal modes 0-3, idler modes 4-7) carrying two
photons in the entangled input state 1/2 sum_k |k>_s |k>_i:

- generator: the 15-phase-shifter mesh of Suppl. Fig. 2(d) on each photon;
- discriminator: the 6-phase-shifter projection circuit of Suppl. Fig. 3 on
  each photon, read at output port 2 of each block.

``pcvl.BS()`` is exactly the chip's multimode interferometer
[[1, i], [i, 1]] / sqrt(2) and ``pcvl.PS`` the phase shifter, so the circuit is
the one simulated by ``lib.circuits``, element for element. All 42 phases are
per-call input parameters, so the QGAN game of ``lib.qgan`` (parameter shift,
noise, counts, stuck phase shifters) drives the layer exactly as it drives the
numpy simulator; ``MerlinHardware`` only replaces the probability computation.

Two noise modes:

- ``injected``: phase noise drawn by ``lib.qgan`` from the per-seed streams, as
  in phase 03 (equivalence check);
- ``phase_error``: MerLin's own noise model, ``pcvl.NoiseModel(phase_error=w)``,
  which adds Uniform(-w, w) to every phase shifter of every execution; w =
  sqrt(3) sigma gives the same standard deviation as the paper's Gaussian.

On a chip, tr[M tau] can only be measured by preparing tau with the generator
meshes: ``target_preparation_phases`` solves the signal mesh for U_tau (idler
mesh at the identity). In ``phase_error`` mode that preparation is noisy too.
"""

from __future__ import annotations

import dataclasses
from importlib.metadata import version

import merlin as ml
import numpy as np
import perceval as pcvl
import torch
from merlin.core import StateVector
from scipy.optimize import least_squares

from .circuits import (
    DISCRIMINATOR_LAYOUT,
    GENERATOR_LAYOUT,
    GENERATOR_NUM_PHASES,
    circuit_unitaries,
    generator_identity_phases,
)
from .data import SeedInputs, seed_stream
from .qgan import QganSettings, _Hardware

NUM_MODES = 8
NUM_PHOTONS = 2
IDLER_OFFSET = 4
NOISE_MODES = ("injected", "phase_error")
# (layout, mode offset, parameter tag) in the order of the 42 per-call phases:
# signal generator, idler generator, signal discriminator, idler discriminator.
CHIP_BLOCKS = (
    (GENERATOR_LAYOUT, 0, "gs"),
    (GENERATOR_LAYOUT, IDLER_OFFSET, "gi"),
    (DISCRIMINATOR_LAYOUT, 0, "ds"),
    (DISCRIMINATOR_LAYOUT, IDLER_OFFSET, "di"),
)


def chip_circuit() -> tuple[pcvl.Circuit, list[str]]:
    """Return the 8-mode chip circuit and its 42 parameter names, in call order."""
    circuit = pcvl.Circuit(NUM_MODES)
    names = []
    for layout, offset, tag in CHIP_BLOCKS:
        for index, (kind, mode) in enumerate(layout):
            name = f"{tag}{index:02d}"
            names.append(name)
            if kind == "phase":
                circuit.add(offset + mode, pcvl.PS(pcvl.P(name)))
            else:
                # MZI(theta) = BS . PS(theta) on the upper arm . BS (Suppl. Eq. 4).
                circuit.add(offset + mode, pcvl.BS())
                circuit.add(offset + mode, pcvl.PS(pcvl.P(name)))
                circuit.add(offset + mode, pcvl.BS())
    return circuit, names


def entangled_input_state() -> StateVector:
    """Return 1/2 sum_k |1 at signal mode k, 1 at idler mode k> (main text Eq. 3)."""
    state = None
    for k in range(4):
        occupation = [0] * NUM_MODES
        occupation[k] = 1
        occupation[IDLER_OFFSET + k] = 1
        term = pcvl.StateVector(pcvl.BasicState(occupation))
        state = term if state is None else state + term
    state.normalize()
    return StateVector.from_perceval(state)


def build_chip_layer(phase_error: float | None = None) -> ml.QuantumLayer:
    """Build the chip as a MerLin layer.

    Parameters
    ----------
    phase_error : float | None
        Half-width w of MerLin's Uniform(-w, w) phase noise on every phase
        shifter; None or 0 gives a noiseless layer. Default value is None.

    Returns
    -------
    merlin.QuantumLayer
        Threshold detection in the ``UNBUNCHED`` space (28 outcomes for 2
        photons in 8 modes), float64, CPU, no trainable parameters.
    """
    circuit, names = chip_circuit()
    noise = pcvl.NoiseModel(phase_error=phase_error) if phase_error else None
    layer = ml.QuantumLayer(
        circuit=circuit,
        input_state=entangled_input_state(),
        n_photons=NUM_PHOTONS,
        input_parameters=names,
        trainable_parameters=[],
        noise=noise,
        n_phase_error_samples=1,
        measurement_strategy=ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.UNBUNCHED
        ),
        dtype=torch.float64,
    )
    layer.eval()
    return layer


def coincidence_index(layer: ml.QuantumLayer) -> np.ndarray:
    """Return ``(4, 4)`` output indices: [a, b] is signal at port a, idler at port b.

    Raises
    ------
    ValueError
        If an output pattern is missing (the layer is not the chip).
    """
    position = {
        tuple(int(n) for n in key): i for i, key in enumerate(layer.output_keys)
    }
    index = np.empty((4, 4), dtype=int)
    for a in range(4):
        for b in range(4):
            occupation = [0] * NUM_MODES
            occupation[a] = 1
            occupation[IDLER_OFFSET + b] = 1
            if tuple(occupation) not in position:
                raise ValueError(f"Output pattern {occupation} missing from the layer")
            index[a, b] = position[tuple(occupation)]
    return index


def chip_probabilities(
    layer: ml.QuantumLayer, phases: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Coincidence probabilities ``(N, 4, 4)`` for phases ``(N, 42)`` in call order."""
    columns = [
        torch.from_numpy(np.ascontiguousarray(phases[:, j : j + 1]))
        for j in range(phases.shape[1])
    ]
    with torch.no_grad():
        probabilities = layer(*columns).numpy()
    return probabilities[:, index]


def target_preparation_phases(
    inputs: SeedInputs, restarts: int = 20, tolerance: float = 1e-24
) -> np.ndarray:
    """Generator phases ``(S, 30)`` that prepare each target on the chip.

    The signal mesh is solved for U_tau up to a global phase by least squares
    (16 unknowns: 15 phases and the global phase) from seeded random starts;
    the idler mesh stays at the identity. Then (U_s x 1)|psi0> = |tau>.

    Raises
    ------
    RuntimeError
        If no start reaches a residual cost below ``tolerance``.
    """
    identity = generator_identity_phases()
    phases = []
    for seed, target in zip(
        inputs.seeds.tolist(), inputs.target_amplitudes, strict=True
    ):
        unitary = 2.0 * target
        rng = seed_stream(seed, "tau_preparation")

        def residual(x: np.ndarray, unitary: np.ndarray = unitary) -> np.ndarray:
            mesh = circuit_unitaries(x[None, :GENERATOR_NUM_PHASES], GENERATOR_LAYOUT)[
                0
            ]
            difference = mesh - np.exp(1j * x[GENERATOR_NUM_PHASES]) * unitary
            return np.concatenate([difference.real.ravel(), difference.imag.ravel()])

        best = None
        for _ in range(restarts):
            start = rng.uniform(0.0, 2.0 * np.pi, GENERATOR_NUM_PHASES + 1)
            solution = least_squares(
                residual, start, xtol=1e-15, ftol=1e-15, gtol=1e-15
            )
            if best is None or solution.cost < best.cost:
                best = solution
            if best.cost < tolerance:
                break
        if best.cost >= tolerance:
            raise RuntimeError(
                f"Could not prepare the target of seed {seed} (cost {best.cost:.2e})"
            )
        phases.append(np.concatenate([best.x[:GENERATOR_NUM_PHASES], identity]))
    return np.stack(phases)


class MerlinHardware(_Hardware):
    """The QGAN chip simulated by a MerLin layer instead of ``lib.circuits``.

    Parameters
    ----------
    inputs : SeedInputs
        Targets and stuck sets of the run.
    settings : QganSettings
        The run's settings; ``phase_sigma`` is the paper's sigma in radians.
    noise : str
        ``injected`` (phase-03 noise model) or ``phase_error`` (MerLin's).
    """

    def __init__(self, inputs: SeedInputs, settings: QganSettings, noise: str) -> None:
        if noise not in NOISE_MODES:
            raise ValueError(f"noise must be one of {NOISE_MODES}, got {noise!r}")
        self.noise_mode = noise
        if noise == "phase_error":
            # MerLin draws the noise inside the layer: nothing is injected here.
            super().__init__(inputs, dataclasses.replace(settings, phase_sigma=0.0))
            self.phase_error = float(np.sqrt(3.0) * settings.phase_sigma)
            if settings.noise_redraw != "execution":
                raise ValueError("MerLin phase_error noise is drawn at every execution")
        else:
            super().__init__(inputs, settings)
            self.phase_error = 0.0
        self.layer = build_chip_layer(self.phase_error)
        self.exact_layer = build_chip_layer(None) if self.phase_error else self.layer
        self.index = coincidence_index(self.layer)
        self.target_phases = target_preparation_phases(inputs)

    def _generated_probabilities(
        self, g_flat: np.ndarray, d_flat: np.ndarray
    ) -> np.ndarray:
        return chip_probabilities(
            self.layer, np.concatenate([g_flat, d_flat], axis=1), self.index
        )

    def _target_probabilities(
        self, rows: np.ndarray, e: int, d_flat: np.ndarray, exact: bool
    ) -> np.ndarray:
        g_flat = np.repeat(self.target_phases[rows], e, axis=0)
        layer = self.exact_layer if exact else self.layer
        return chip_probabilities(
            layer, np.concatenate([g_flat, d_flat], axis=1), self.index
        )

    def hardware_fields(self, counts: int | None) -> dict:
        """Hardware-aware description of the runs (Phase-4 reporting contract)."""
        return {
            # merlinquantum 0.4.2 still reports merlin.__version__ == "0.4.1", so
            # record the installed distribution as well.
            "framework": (
                f"merlinquantum {version('merlinquantum')} (merlin.__version__ "
                f"{ml.__version__}), perceval-quandela {version('perceval-quandela')}"
            ),
            "computation_space": "UNBUNCHED",
            "detector_model": "threshold (at most one photon per 4-mode block, so equal to number resolution here)",
            "photon_number": NUM_PHOTONS,
            "number_of_modes": NUM_MODES,
            "input_state": "1/2 (|10001000> + |01000100> + |00100010> + |00010001>) (signal modes 0-3, idler 4-7)",
            "encoding": "42 phase shifters as per-call input parameters (30 generator, 12 discriminator)",
            "measurement_strategy": "MeasurementStrategy.probs(computation_space=UNBUNCHED); tr[M rho] = P(signal port 2, idler port 2)",
            "postselection": "one photon per block; no probability leaves the 16 coincidence patterns (block-diagonal circuit)",
            "simulator": "MerLin CPU analytic simulator (float64); counts drawn from its exact probabilities",
            "shot_count": counts,
            "noise_mode": self.noise_mode,
            "merlin_phase_error_half_width_rad": self.phase_error,
            "tau_measurement": "tau prepared on the chip by the generator meshes (phases solved by least squares)",
            "gradient": "parameter shift (pi/4), each shifted term a separate noisy execution",
        }


__all__ = [
    "MerlinHardware",
    "build_chip_layer",
    "chip_circuit",
    "chip_probabilities",
    "coincidence_index",
    "entangled_input_state",
    "target_preparation_phases",
]
