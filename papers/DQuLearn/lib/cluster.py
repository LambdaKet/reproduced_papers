"""Local DQuLearn deployment: one process per quantum worker, reached over RPyC.

Stands in for the paper's Google Cloud deployment (Sec. IV-C2), where each
worker was an e2-medium VM: here each worker is a process on this machine with
a fixed number of simulation slots, reached over a localhost RPyC connection.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rpyc

from .comanager import CoManager
from .execution import CircuitTask, CircuitTiming, ExecutionSettings

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkerSpec:
    """One worker to start.

    Parameters
    ----------
    max_qubits : int
        Maximum qubits the worker holds at once (MR).
    cpu_slots : int, optional
        Circuits it simulates at the same time. Default value is 1.
    """

    max_qubits: int
    cpu_slots: int = 1


class RpycWorkerHandle:
    """Manager-side handle of one worker.

    Holds one RPyC connection per circuit the worker may run at once, so that
    concurrent circuits are served in parallel by the worker's threaded server,
    plus one connection for registration and heartbeats.

    Parameters
    ----------
    name : str
        Worker name.
    port : int
        Localhost port of the worker's RPyC server.
    connections : int
        Connections opened for circuit execution.
    call_timeout : float
        Timeout of an execution call, in seconds.
    status_timeout : float
        Timeout of a registration or heartbeat call, in seconds.
    """

    def __init__(
        self,
        name: str,
        port: int,
        connections: int,
        call_timeout: float,
        status_timeout: float,
    ) -> None:
        self.name = name
        self.port = port
        self._call_connections: queue.Queue = queue.Queue()
        for _ in range(connections):
            self._call_connections.put(self._connect(call_timeout))
        self._status_connection = self._connect(status_timeout)
        self._status_lock = threading.Lock()

    def _connect(self, timeout: float) -> rpyc.Connection:
        return rpyc.connect(
            "127.0.0.1", self.port, config={"sync_request_timeout": timeout}
        )

    def register(self) -> dict:
        with self._status_lock:
            return json.loads(self._status_connection.root.register())

    def status(self) -> dict:
        with self._status_lock:
            return json.loads(self._status_connection.root.status())

    def execute(self, task: CircuitTask) -> tuple[float, float, float]:
        connection = self._call_connections.get()
        try:
            value, worker_wait, execution = connection.root.execute(
                task.circuit, task.qubits, task.seed
            )
        finally:
            self._call_connections.put(connection)
        return float(value), float(worker_wait), float(execution)

    def close(self) -> None:
        while not self._call_connections.empty():
            self._call_connections.get_nowait().close()
        self._status_connection.close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LocalCluster:
    """Starts worker processes and connects to them; use as a context manager.

    Parameters
    ----------
    specs : list of WorkerSpec
        Workers to start, in registration order.
    settings : ExecutionSettings
        Shots, exact mode and compile mode passed to every worker.
    log_dir : pathlib.Path
        Folder receiving ``worker-<i>.log`` and ``worker-<i>.stderr.log``.
    connections_per_worker : list of int
        Execution connections per worker: the most circuits it may run at once.
    startup_timeout : float, optional
        Seconds to wait for each worker to accept connections. Default value
        is 120.0.
    call_timeout : float, optional
        Timeout of an execution call. Default value is 600.0.
    status_timeout : float, optional
        Timeout of a heartbeat call. Default value is 5.0.
    """

    def __init__(
        self,
        specs: list[WorkerSpec],
        settings: ExecutionSettings,
        log_dir: Path,
        *,
        connections_per_worker: list[int],
        startup_timeout: float = 120.0,
        call_timeout: float = 600.0,
        status_timeout: float = 5.0,
    ) -> None:
        if len(connections_per_worker) != len(specs):
            raise ValueError("one connection count per worker is required")
        self.specs = specs
        self.settings = settings
        self.log_dir = Path(log_dir)
        self.connections_per_worker = connections_per_worker
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self.status_timeout = status_timeout
        self.handles: list[RpycWorkerHandle] = []
        self.startup_seconds = 0.0
        self._processes: list[tuple[str, int, subprocess.Popen]] = []
        self._stderr_files = []

    def __enter__(self) -> LocalCluster:
        start = time.perf_counter()
        try:
            for index, spec in enumerate(self.specs):
                self._start_worker(f"worker-{index + 1}", spec)
            for (name, port, process), connections in zip(
                self._processes, self.connections_per_worker, strict=True
            ):
                self._wait_until_ready(name, port, process)
                self.handles.append(
                    RpycWorkerHandle(
                        name, port, connections, self.call_timeout, self.status_timeout
                    )
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.startup_seconds = time.perf_counter() - start
        logger.info(
            "CLUSTER_READY | workers=%d | max_qubits=%s | cpu_slots=%s | seconds=%.2f",
            len(self.specs),
            [spec.max_qubits for spec in self.specs],
            [spec.cpu_slots for spec in self.specs],
            self.startup_seconds,
        )
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self.handles:
            handle.close()
        for _, _, process in self._processes:
            process.terminate()
        for name, _, process in self._processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("WORKER_KILLED | worker=%s", name)
                process.kill()
                process.wait()
        for handle in self._stderr_files:
            handle.close()
        self.handles = []
        self._processes = []
        self._stderr_files = []

    def _start_worker(self, name: str, spec: WorkerSpec) -> None:
        port = _free_port()
        command = [
            sys.executable,
            "-m",
            "lib.worker",
            "--name",
            name,
            "--port",
            str(port),
            "--max-qubits",
            str(spec.max_qubits),
            "--cpu-slots",
            str(spec.cpu_slots),
            "--shots",
            str(self.settings.shots),
            "--compile-mode",
            self.settings.compile_mode,
            "--log-file",
            str(self.log_dir / f"{name}.log"),
        ]
        if self.settings.exact:
            command.append("--exact")
        stderr_file = (self.log_dir / f"{name}.stderr.log").open("w", encoding="utf-8")
        self._stderr_files.append(stderr_file)
        # "-m lib.worker" from the paper folder: the worker imports this package.
        process = subprocess.Popen(
            command, cwd=PROJECT_DIR, stdout=subprocess.DEVNULL, stderr=stderr_file
        )
        self._processes.append((name, port, process))

    def _wait_until_ready(
        self, name: str, port: int, process: subprocess.Popen
    ) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{name} exited with code {process.returncode}; see "
                    f"{self.log_dir / f'{name}.stderr.log'}"
                )
            try:
                rpyc.connect("127.0.0.1", port).close()
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{name} did not accept connections within "
                        f"{self.startup_timeout:.0f} s"
                    ) from None
                time.sleep(0.1)


class ManagerExecutor:
    """Runs the circuit banks of one client (one training job) through the co-Manager.

    Parameters
    ----------
    manager : CoManager
        Shared co-Manager; the cluster owns it and its workers.
    client : str
        Client name recorded with every assignment.
    """

    def __init__(self, manager: CoManager, client: str) -> None:
        self.manager = manager
        self.client = client
        self.timings: list[CircuitTiming] = []

    def run(self, tasks: list[CircuitTask]) -> list[float]:
        results = self.manager.run_bank(tasks, self.client)
        self.timings.extend(result.timing for result in results)
        return [result.value for result in results]

    def close(self) -> None:
        """The cluster owns the manager and the workers."""
