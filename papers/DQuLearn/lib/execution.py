"""Per-circuit work shared by the quantum workers and the execution baselines.

The DQuLearn workers, the plain process pool and the local reference all run
:class:`CircuitRunner`, so they differ only in how circuits reach it. The
batched baseline compiles and runs a whole circuit bank as one Aer job, the
usual way to execute many small circuits.

A runner does what a DQuLearn worker does with a logical circuit (Fig. 1):
deserialize it, map it onto the simulator (``transpile``, the Quantum Data
Loader), run it (Circuit Executor) and turn the ancilla counts into a fidelity
(Quantum Measurement).
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from .circuits import circuit_from_bytes, fidelity_from_probability

# "per_circuit" calls transpile() for every circuit, rebuilding the compiler each
# time, as QuClassi's code does through qiskit.execute(); "cached" builds the
# compiler once per runner. Both use optimization level 2, transpile's default.
COMPILE_MODES = ("per_circuit", "cached")
OPTIMIZATION_LEVEL = 2


def quiet_qiskit_logging() -> None:
    """Keep Qiskit's records at WARNING and above.

    At INFO, Qiskit's compiler logs every pass, about 44 lines per circuit,
    which would bury the experiment events in ``run.log`` and the worker logs.
    """
    logging.getLogger("qiskit").setLevel(logging.WARNING)


@dataclass(frozen=True)
class CircuitTask:
    """One logical circuit to execute.

    Parameters
    ----------
    circuit : bytes
        QPY serialization of the circuit.
    qubits : int
        Circuit width, the qubit demand the co-Manager checks against a worker.
    seed : int
        Simulator seed for this circuit's shots, so a circuit gives the same
        counts whichever worker runs it.
    """

    circuit: bytes
    qubits: int
    seed: int


@dataclass(frozen=True)
class ExecutionSettings:
    """How circuits are executed.

    Parameters
    ----------
    shots : int
        Shots per circuit when sampling.
    exact : bool
        Use the exact ancilla probability from the statevector instead of shots.
    compile_mode : str
        ``per_circuit`` or ``cached`` (see ``COMPILE_MODES``).

    Raises
    ------
    ValueError
        If the compile mode is unknown or shots are not positive when sampling.
    """

    shots: int
    exact: bool
    compile_mode: str

    def __post_init__(self) -> None:
        if self.compile_mode not in COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {COMPILE_MODES}, got {self.compile_mode!r}"
            )
        if not self.exact and self.shots <= 0:
            raise ValueError(f"shots must be positive when sampling, got {self.shots}")


@dataclass(frozen=True)
class CircuitTiming:
    """Timing of one executed circuit, in seconds.

    Parameters
    ----------
    worker : str
        Name of the worker, process or executor that ran the circuit.
    queued : float
        From submission to assignment to a worker (co-Manager only).
    round_trip : float
        From assignment to the result being back (equals ``execution`` locally).
    worker_wait : float
        Time spent inside the worker waiting for a free simulation slot.
    execution : float
        Deserialization, compilation, simulation and read-out.
    """

    worker: str
    queued: float
    round_trip: float
    worker_wait: float
    execution: float


class CircuitRunner:
    """Executes logical circuits one at a time in the current process.

    Parameters
    ----------
    settings : ExecutionSettings
        Shots, exact mode and compile mode.
    """

    def __init__(self, settings: ExecutionSettings) -> None:
        self.settings = settings
        self._simulator = AerSimulator(max_parallel_threads=1)
        self._pass_manager = (
            generate_preset_pass_manager(
                optimization_level=OPTIMIZATION_LEVEL, backend=self._simulator
            )
            if settings.compile_mode == "cached"
            else None
        )

    def run(self, circuit_bytes: bytes, seed: int) -> float:
        """Execute one serialized circuit and return its fidelity estimate.

        Parameters
        ----------
        circuit_bytes : bytes
            QPY serialization of a swap-test circuit.
        seed : int
            Simulator seed for the shots (unused in exact mode).

        Returns
        -------
        float
            Fidelity estimate ``max(2·P(0) − 1, 1e-16)``.
        """
        circuit = circuit_from_bytes(circuit_bytes)
        if self.settings.exact:
            state = Statevector(circuit.remove_final_measurements(inplace=False))
            probability_zero = float(state.probabilities([0])[0])
        else:
            if self._pass_manager is None:
                compiled = transpile(
                    circuit, self._simulator, optimization_level=OPTIMIZATION_LEVEL
                )
            else:
                compiled = self._pass_manager.run(circuit)
            counts = (
                self._simulator.run(
                    compiled, shots=self.settings.shots, seed_simulator=seed
                )
                .result()
                .get_counts()
            )
            probability_zero = counts.get("0", 0) / self.settings.shots
        return fidelity_from_probability(probability_zero)


class LocalExecutor:
    """Runs every circuit in this process, one after another: no distribution.

    Parameters
    ----------
    settings : ExecutionSettings
        Shots, exact mode and compile mode.
    """

    def __init__(self, settings: ExecutionSettings) -> None:
        self._runner = CircuitRunner(settings)
        self.timings: list[CircuitTiming] = []

    def run(self, tasks: list[CircuitTask]) -> list[float]:
        values = []
        for task in tasks:
            start = time.perf_counter()
            values.append(self._runner.run(task.circuit, task.seed))
            elapsed = time.perf_counter() - start
            self.timings.append(CircuitTiming("local", 0.0, elapsed, 0.0, elapsed))
        return values

    def close(self) -> None:
        """Nothing to release."""


class BatchedExecutor:
    """Runs each circuit bank as a single Aer job in this process.

    The bank is compiled with one compiler build and simulated in one job. A
    batch uses one simulator seed (the first task's), so its shot noise differs
    from per-circuit execution while following the same distribution.

    Parameters
    ----------
    settings : ExecutionSettings
        Shots, exact mode and compile mode (the compile mode does not apply).
    """

    def __init__(self, settings: ExecutionSettings) -> None:
        self.settings = settings
        self._runner = CircuitRunner(settings)
        self._simulator = AerSimulator(max_parallel_threads=1)
        self.timings: list[CircuitTiming] = []

    def run(self, tasks: list[CircuitTask]) -> list[float]:
        start = time.perf_counter()
        if self.settings.exact:
            values = [self._runner.run(task.circuit, task.seed) for task in tasks]
        else:
            circuits = [circuit_from_bytes(task.circuit) for task in tasks]
            compiled = transpile(
                circuits,
                self._simulator,
                optimization_level=OPTIMIZATION_LEVEL,
                num_processes=1,
            )
            result = self._simulator.run(
                compiled, shots=self.settings.shots, seed_simulator=tasks[0].seed
            ).result()
            values = [
                fidelity_from_probability(
                    result.get_counts(index).get("0", 0) / self.settings.shots
                )
                for index in range(len(tasks))
            ]
        per_circuit = (time.perf_counter() - start) / max(1, len(tasks))
        self.timings.extend(
            CircuitTiming("batched", 0.0, per_circuit, 0.0, per_circuit) for _ in tasks
        )
        return values

    def close(self) -> None:
        """Nothing to release."""


_POOL_RUNNER: CircuitRunner | None = None


def _initialize_pool_runner(settings: ExecutionSettings) -> None:
    global _POOL_RUNNER
    _POOL_RUNNER = CircuitRunner(settings)


def _run_in_pool(circuit_bytes: bytes, seed: int) -> tuple[float, float]:
    if _POOL_RUNNER is None:
        raise RuntimeError("pool process was not initialized")
    start = time.perf_counter()
    value = _POOL_RUNNER.run(circuit_bytes, seed)
    return value, time.perf_counter() - start


class ProcessPoolCircuitExecutor:
    """Runs circuits on a plain pool of processes: no manager, RPC or heartbeats.

    Parameters
    ----------
    settings : ExecutionSettings
        Shots, exact mode and compile mode.
    processes : int
        Number of worker processes.
    """

    def __init__(self, settings: ExecutionSettings, processes: int) -> None:
        if processes < 1:
            raise ValueError(f"processes must be >= 1, got {processes}")
        self._pool = ProcessPoolExecutor(
            max_workers=processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_pool_runner,
            initargs=(settings,),
        )
        self.timings: list[CircuitTiming] = []

    def run(self, tasks: list[CircuitTask]) -> list[float]:
        futures = [
            self._pool.submit(_run_in_pool, task.circuit, task.seed) for task in tasks
        ]
        values = []
        for future in futures:
            value, execution = future.result()
            values.append(value)
            # Only the execution time is known per circuit; the bank's wall-clock
            # is measured by the caller (epoch timer).
            self.timings.append(CircuitTiming("pool", 0.0, execution, 0.0, execution))
        return values

    def close(self) -> None:
        self._pool.shutdown(wait=True)
