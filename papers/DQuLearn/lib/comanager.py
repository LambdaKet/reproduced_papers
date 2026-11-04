"""co-Manager of DQuLearn (Algorithm 2, Sec. III-C).

For each registered quantum worker the co-Manager keeps its maximum qubits MR,
occupied qubits OR, available qubits AR = MR − OR and classical resource usage
CRU. A worker registers with OR = 0 and AR = MR (lines 2–6); periodic heartbeats
refresh CRU, and a worker that misses three in a row is removed (lines 7–13); a
pending circuit goes to the worker with the lowest CRU among those that can hold
it (lines 14–20).

Implementation choices (recorded in LOG.md):

- A worker can hold a circuit when AR >= demand. Algorithm 2 writes AR > demand,
  which would never place a 5-qubit circuit on a 5-qubit worker, whereas the
  workers of Sec. IV-C are configured with 5 qubits to run 5-qubit circuits.
- OR changes when a circuit is assigned and when its result returns, since the
  co-Manager knows what it sent; heartbeats refresh CRU and liveness, and the
  OR they report is kept for diagnostics.
- Waiting circuits are assigned first-fit in arrival order: whenever qubits free
  up, the oldest waiting circuit that fits goes first.
- ``single`` tenancy lets a worker run one circuit at a time; ``multi`` lets a
  worker run as many as its qubits hold.
- ``round_robin`` replaces the lowest-CRU rule by a rotation over the workers
  that can hold the circuit (a baseline added by this reproduction).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from .execution import CircuitTask, CircuitTiming

logger = logging.getLogger(__name__)

POLICIES = ("least_cpu", "round_robin")
TENANCIES = ("multi", "single")


class WorkerHandle(Protocol):
    """What the co-Manager needs from a quantum worker."""

    name: str

    def register(self) -> dict: ...

    def status(self) -> dict: ...

    def execute(self, task: CircuitTask) -> tuple[float, float, float]: ...


@dataclass(eq=False)
class WorkerRecord:
    """The co-Manager's view of one worker (Algorithm 2, line 1)."""

    handle: WorkerHandle
    max_qubits: int
    cpu_usage: float
    occupied_qubits: int = 0
    running_circuits: int = 0
    reported_occupied_qubits: int = 0
    missed_heartbeats: int = 0
    active: bool = True
    assigned_by_client: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.handle.name

    @property
    def available_qubits(self) -> int:
        return self.max_qubits - self.occupied_qubits


@dataclass(frozen=True)
class CircuitResult:
    """Result of one circuit executed through the co-Manager."""

    value: float
    client: str
    timing: CircuitTiming


@dataclass
class _PendingCircuit:
    task: CircuitTask
    client: str
    future: Future
    submitted: float


class CoManager:
    """Assigns circuits from any number of clients to registered workers.

    Parameters
    ----------
    handles : list of WorkerHandle
        Workers to register at initialization.
    policy : str, optional
        ``least_cpu`` (Algorithm 2) or ``round_robin``. Default value is
        ``least_cpu``.
    tenancy : str, optional
        ``multi`` or ``single``. Default value is ``multi``.
    heartbeat_seconds : float, optional
        Heartbeat period; the paper uses 5 s. Default value is 5.0.
    missed_heartbeat_limit : int, optional
        Consecutive missed heartbeats after which a worker is removed.
        Default value is 3.
    max_parallel_calls : int, optional
        Threads available for in-flight circuits. Default value is 64.

    Raises
    ------
    ValueError
        If a policy, tenancy or period is invalid.
    """

    def __init__(
        self,
        handles: list[WorkerHandle],
        *,
        policy: str = "least_cpu",
        tenancy: str = "multi",
        heartbeat_seconds: float = 5.0,
        missed_heartbeat_limit: int = 3,
        max_parallel_calls: int = 64,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        if tenancy not in TENANCIES:
            raise ValueError(f"tenancy must be one of {TENANCIES}, got {tenancy!r}")
        if heartbeat_seconds <= 0 or missed_heartbeat_limit < 1:
            raise ValueError("heartbeat period must be > 0 and the limit >= 1")
        self.policy = policy
        self.tenancy = tenancy
        self.heartbeat_seconds = heartbeat_seconds
        self.missed_heartbeat_limit = missed_heartbeat_limit
        self.heartbeat_rounds = 0
        self._condition = threading.Condition()
        self._workers: list[WorkerRecord] = []
        self._pending: list[_PendingCircuit] = []
        self._next_round_robin = 0
        self._calls = ThreadPoolExecutor(
            max_workers=max_parallel_calls, thread_name_prefix="co-manager-call"
        )
        for handle in handles:
            self.register_worker(handle)
        self._stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="co-manager-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def __enter__(self) -> CoManager:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def register_worker(self, handle: WorkerHandle) -> None:
        """New worker registration (Algorithm 2, lines 2–6): OR = 0, AR = MR."""
        info = handle.register()
        status = handle.status()
        record = WorkerRecord(
            handle=handle,
            max_qubits=int(info["max_qubits"]),
            cpu_usage=float(status["cpu_percent"]),
        )
        with self._condition:
            self._workers.append(record)
            self._assign_waiting_locked()
        logger.info(
            "WORKER_JOINED | worker=%s | max_qubits=%d", record.name, record.max_qubits
        )

    def submit(self, task: CircuitTask, client: str) -> Future:
        """Queue one circuit for assignment and return a future of its result.

        Raises
        ------
        ValueError
            If no active worker can ever hold a circuit of this width.
        """
        future: Future = Future()
        with self._condition:
            if not self._can_ever_hold_locked(task.qubits):
                raise ValueError(
                    f"no active worker can hold a {task.qubits}-qubit circuit"
                )
            self._pending.append(
                _PendingCircuit(task, client, future, time.perf_counter())
            )
            self._assign_waiting_locked()
        return future

    def run_bank(self, tasks: list[CircuitTask], client: str) -> list[CircuitResult]:
        """Submit a circuit bank and wait for all its results, in order."""
        futures = [self.submit(task, client) for task in tasks]
        return [future.result() for future in futures]

    def heartbeat(self) -> None:
        """One round of periodic heartbeats (Algorithm 2, lines 7–13)."""
        for record in list(self._workers):
            if not record.active:
                continue
            try:
                status = record.handle.status()
            except Exception as error:  # noqa: BLE001 - a lost worker is expected here
                self._record_missed_heartbeat(record, error)
                continue
            with self._condition:
                record.missed_heartbeats = 0
                record.cpu_usage = float(status["cpu_percent"])
                record.reported_occupied_qubits = sum(
                    int(demand) for demand in status["active_demands"]
                )
        with self._condition:
            self.heartbeat_rounds += 1

    def reset_assignment_counts(self) -> None:
        """Forget assignment counts so far, e.g. after warm-up circuits."""
        with self._condition:
            for record in self._workers:
                record.assigned_by_client.clear()

    def snapshot(self) -> list[dict]:
        """Per-worker state and assignment counts, for metrics."""
        with self._condition:
            return [
                {
                    "worker": record.name,
                    "max_qubits": record.max_qubits,
                    "active": record.active,
                    "cpu_usage_last": record.cpu_usage,
                    "assigned_by_client": dict(record.assigned_by_client),
                    "assigned_total": sum(record.assigned_by_client.values()),
                }
                for record in self._workers
            ]

    def close(self) -> None:
        """Stop heartbeats and wait for in-flight circuits."""
        self._stop.set()
        self._heartbeat_thread.join()
        self._calls.shutdown(wait=True)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.heartbeat()

    def _record_missed_heartbeat(self, record: WorkerRecord, error: Exception) -> None:
        with self._condition:
            record.missed_heartbeats += 1
            if record.missed_heartbeats < self.missed_heartbeat_limit:
                return
            record.active = False
            stranded = [
                pending
                for pending in self._pending
                if not self._can_ever_hold_locked(pending.task.qubits)
            ]
            for pending in stranded:
                self._pending.remove(pending)
                pending.future.set_exception(
                    RuntimeError(
                        f"worker {record.name} was removed and no remaining worker "
                        f"can hold a {pending.task.qubits}-qubit circuit"
                    )
                )
        logger.warning(
            "WORKER_REMOVED | worker=%s | missed_heartbeats=%d | last_error=%s",
            record.name,
            record.missed_heartbeats,
            error,
        )

    def _can_ever_hold_locked(self, qubits: int) -> bool:
        return any(
            record.active and record.max_qubits >= qubits for record in self._workers
        )

    def _fits(self, record: WorkerRecord, qubits: int) -> bool:
        if not record.active or record.max_qubits < qubits:
            return False
        if self.tenancy == "single":
            return record.running_circuits == 0
        return record.available_qubits >= qubits

    def _select_locked(self, qubits: int) -> WorkerRecord | None:
        candidates = [record for record in self._workers if self._fits(record, qubits)]
        if not candidates:
            return None
        if self.policy == "least_cpu":
            # Algorithm 2, lines 18–20: lowest CRU first; ties keep registration order.
            return min(candidates, key=lambda record: record.cpu_usage)
        count = len(self._workers)
        for offset in range(count):
            position = (self._next_round_robin + offset) % count
            record = self._workers[position]
            if any(record is candidate for candidate in candidates):
                self._next_round_robin = (position + 1) % count
                return record
        return None

    def _assign_waiting_locked(self) -> None:
        index = 0
        while index < len(self._pending):
            pending = self._pending[index]
            record = self._select_locked(pending.task.qubits)
            if record is None:
                index += 1
                continue
            del self._pending[index]
            record.occupied_qubits += pending.task.qubits
            record.running_circuits += 1
            record.assigned_by_client[pending.client] = (
                record.assigned_by_client.get(pending.client, 0) + 1
            )
            self._calls.submit(self._execute, record, pending, time.perf_counter())

    def _execute(
        self, record: WorkerRecord, pending: _PendingCircuit, assigned_at: float
    ) -> None:
        result, failure = None, None
        try:
            value, worker_wait, execution = record.handle.execute(pending.task)
            timing = CircuitTiming(
                worker=record.name,
                queued=assigned_at - pending.submitted,
                round_trip=time.perf_counter() - assigned_at,
                worker_wait=worker_wait,
                execution=execution,
            )
            result = CircuitResult(value, pending.client, timing)
        except Exception as error:  # noqa: BLE001 - handed to the waiting client
            failure = error
        # Free the worker before the client sees the result, so a client that
        # resubmits at once finds the worker available.
        with self._condition:
            record.occupied_qubits -= pending.task.qubits
            record.running_circuits -= 1
            self._assign_waiting_locked()
        if failure is None:
            pending.future.set_result(result)
        else:
            pending.future.set_exception(failure)
