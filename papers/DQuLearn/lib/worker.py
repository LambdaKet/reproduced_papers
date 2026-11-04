"""DQuLearn quantum worker: an RPyC service that executes logical circuits.

The worker side of DQuLearn's Fig. 1 and Algorithm 2. It reports its maximum
qubits when the co-Manager registers it, its active circuits and classical
resource usage at every heartbeat, and executes the circuits assigned to it.
Each worker runs in its own process with a fixed number of simulation slots,
its CPU budget (standing in for the 1–2 vCPUs of the paper's e2-medium VMs);
circuits beyond that number wait inside the worker.

:mod:`lib.cluster` starts it as ``python -m lib.worker`` from the paper folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time

import psutil
import rpyc
from rpyc.utils.server import ThreadedServer

from .execution import CircuitRunner, ExecutionSettings, quiet_qiskit_logging
from .experiment_logging import configure_logging

logger = logging.getLogger(__name__)


class QuantumWorkerService(rpyc.Service):
    """RPyC service exposing registration, heartbeat status and circuit execution.

    Parameters
    ----------
    name : str
        Worker name used in logs and assignment counts.
    max_qubits : int
        Maximum qubits the worker can hold at once (MR in Algorithm 2).
    cpu_slots : int
        Circuits simulated at the same time; further assigned circuits wait.
    settings : ExecutionSettings
        Shots, exact mode and compile mode.
    """

    def __init__(
        self, name: str, max_qubits: int, cpu_slots: int, settings: ExecutionSettings
    ) -> None:
        super().__init__()
        self._name = name
        self._max_qubits = max_qubits
        self._cpu_slots = cpu_slots
        self._settings = settings
        self._slots = threading.BoundedSemaphore(cpu_slots)
        self._lock = threading.Lock()
        self._active_demands: dict[int, int] = {}
        self._next_ticket = 0
        self._executed = 0
        self._process = psutil.Process()
        self._process.cpu_percent(None)  # psutil's first call only primes the counter
        self._runners = threading.local()

    def exposed_register(self) -> str:
        """Return the worker's identity and maximum qubits, as JSON."""
        info = {
            "name": self._name,
            "max_qubits": self._max_qubits,
            "cpu_slots": self._cpu_slots,
            "pid": os.getpid(),
        }
        logger.info("WORKER_REGISTERED | %s", info)
        return json.dumps(info)

    def exposed_status(self) -> str:
        """Return the heartbeat payload as JSON.

        ``active_demands`` lists the qubit demand of every circuit assigned and
        not finished (ACw in Algorithm 2); ``cpu_percent`` is this process's CPU
        usage since the previous heartbeat (CRUw).
        """
        with self._lock:
            active = list(self._active_demands.values())
            executed = self._executed
        status = {
            "active_demands": active,
            "cpu_percent": self._process.cpu_percent(None),
            "executed": executed,
        }
        logger.debug("HEARTBEAT | %s", status)
        return json.dumps(status)

    def exposed_execute(
        self, circuit: bytes, qubits: int, seed: int
    ) -> tuple[float, float, float]:
        """Execute one QPY circuit.

        Returns
        -------
        tuple of float
            Fidelity estimate, seconds waited for a simulation slot, and seconds
            spent executing (deserialize, compile, simulate, read out).
        """
        received = time.perf_counter()
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._active_demands[ticket] = int(qubits)
        try:
            with self._slots:
                started = time.perf_counter()
                fidelity = self._runner().run(bytes(circuit), int(seed))
                finished = time.perf_counter()
        finally:
            with self._lock:
                del self._active_demands[ticket]
                self._executed += 1
        return fidelity, started - received, finished - started

    def _runner(self) -> CircuitRunner:
        # One runner per server thread: a cached compiler is not shared across threads.
        runner = getattr(self._runners, "runner", None)
        if runner is None:
            runner = CircuitRunner(self._settings)
            self._runners.runner = runner
        return runner


def main(argv: list[str] | None = None) -> int:
    """Start a worker and serve until terminated."""
    parser = argparse.ArgumentParser(description="DQuLearn quantum worker (RPyC).")
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-qubits", type=int, required=True)
    parser.add_argument("--cpu-slots", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--compile-mode", required=True)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args(argv)

    configure_logging(args.log_file, console=False)
    quiet_qiskit_logging()
    settings = ExecutionSettings(
        shots=args.shots, exact=args.exact, compile_mode=args.compile_mode
    )
    service = QuantumWorkerService(args.name, args.max_qubits, args.cpu_slots, settings)
    server = ThreadedServer(
        service,
        hostname="127.0.0.1",
        port=args.port,
        protocol_config={"allow_public_attrs": False, "sync_request_timeout": None},
    )
    signal.signal(signal.SIGTERM, lambda *_: server.close())
    logger.info(
        "WORKER_STARTED | name=%s | port=%d | max_qubits=%d | cpu_slots=%d | pid=%d",
        args.name,
        args.port,
        args.max_qubits,
        args.cpu_slots,
        os.getpid(),
    )
    server.start()
    logger.info("WORKER_STOPPED | name=%s", args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
