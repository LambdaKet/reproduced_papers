from __future__ import annotations

import threading
import time

import pytest
from common import (
    PROJECT_DIR,  # noqa: F401  (puts runtime_lib and the DQuLearn package on sys.path)
)
from DQuLearn.lib.comanager import CoManager
from DQuLearn.lib.execution import CircuitTask

NO_HEARTBEAT = 3600.0  # tests trigger heartbeats by hand


class FakeWorker:
    """In-process stand-in for a quantum worker: returns the task's seed."""

    def __init__(
        self, name: str, max_qubits: int, cpu: float = 0.0, delay: float = 0.0
    ):
        self.name = name
        self.max_qubits = max_qubits
        self.cpu = cpu
        self.delay = delay
        self.alive = True
        self.executed: list[int] = []
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()

    def register(self) -> dict:
        return {"max_qubits": self.max_qubits}

    def status(self) -> dict:
        if not self.alive:
            raise ConnectionError("worker down")
        return {"active_demands": [], "cpu_percent": self.cpu}

    def execute(self, task: CircuitTask) -> tuple[float, float, float]:
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(self.delay)
        with self._lock:
            self.running -= 1
            self.executed.append(task.seed)
        return float(task.seed), 0.0, self.delay


def _tasks(qubits: int, count: int) -> list[CircuitTask]:
    return [CircuitTask(b"", qubits, seed) for seed in range(count)]


def test_lowest_cpu_worker_that_fits_gets_the_circuit():
    busy, idle = FakeWorker("busy", 5, cpu=50.0), FakeWorker("idle", 5, cpu=10.0)
    with CoManager([busy, idle], heartbeat_seconds=NO_HEARTBEAT) as manager:
        manager.run_bank(_tasks(5, 1), "client")
    assert idle.executed == [0] and busy.executed == []


def test_five_qubit_worker_runs_a_five_qubit_circuit():
    # Algorithm 2 writes AR > demand; with ">" this circuit could never run.
    worker = FakeWorker("w", 5)
    with CoManager([worker], heartbeat_seconds=NO_HEARTBEAT) as manager:
        results = manager.run_bank(_tasks(5, 3), "client")
    assert [result.value for result in results] == [0.0, 1.0, 2.0]


def test_seven_qubit_circuit_skips_the_five_qubit_worker():
    small, large = FakeWorker("small", 5, cpu=0.0), FakeWorker("large", 10, cpu=90.0)
    with CoManager([small, large], heartbeat_seconds=NO_HEARTBEAT) as manager:
        manager.run_bank(_tasks(7, 2), "client")
    assert small.executed == [] and sorted(large.executed) == [0, 1]


@pytest.mark.parametrize(
    ("tenancy", "expected_max_running"), [("multi", 4), ("single", 1)]
)
def test_tenancy_sets_how_many_circuits_share_a_worker(tenancy, expected_max_running):
    worker = FakeWorker("w", 20, delay=0.05)
    with CoManager(
        [worker], tenancy=tenancy, heartbeat_seconds=NO_HEARTBEAT
    ) as manager:
        manager.run_bank(_tasks(5, 8), "client")
    assert worker.max_running == expected_max_running


def test_round_robin_rotates_over_workers():
    first, second = FakeWorker("first", 5, cpu=0.0), FakeWorker("second", 5, cpu=99.0)
    with CoManager(
        [first, second], policy="round_robin", heartbeat_seconds=NO_HEARTBEAT
    ) as manager:
        for seed in range(4):
            manager.run_bank([CircuitTask(b"", 5, seed)], "client")
    assert first.executed == [0, 2] and second.executed == [1, 3]


def test_worker_is_removed_after_three_missed_heartbeats():
    healthy, failing = FakeWorker("healthy", 5), FakeWorker("failing", 10)
    with CoManager([healthy, failing], heartbeat_seconds=NO_HEARTBEAT) as manager:
        failing.alive = False
        manager.heartbeat()
        manager.heartbeat()
        assert all(state["active"] for state in manager.snapshot())
        manager.heartbeat()
        states = {state["worker"]: state["active"] for state in manager.snapshot()}
        assert states == {"healthy": True, "failing": False}
        with pytest.raises(ValueError):
            manager.submit(CircuitTask(b"", 7, 0), "client")


def test_heartbeat_refreshes_cpu_usage():
    first, second = FakeWorker("first", 5, cpu=0.0), FakeWorker("second", 5, cpu=50.0)
    with CoManager([first, second], heartbeat_seconds=NO_HEARTBEAT) as manager:
        first.cpu, second.cpu = 80.0, 5.0
        manager.heartbeat()
        manager.run_bank(_tasks(5, 1), "client")
    assert second.executed == [0]


def test_circuit_too_wide_for_every_worker_is_rejected():
    with CoManager([FakeWorker("w", 5)], heartbeat_seconds=NO_HEARTBEAT) as manager:
        with pytest.raises(ValueError):
            manager.submit(CircuitTask(b"", 7, 0), "client")
