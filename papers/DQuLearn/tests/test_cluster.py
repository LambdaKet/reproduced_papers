from __future__ import annotations

import time

import numpy as np
import pytest
from common import (
    PROJECT_DIR,  # noqa: F401  (puts runtime_lib and the DQuLearn package on sys.path)
)
from DQuLearn.lib.circuits import (
    QuClassiLayout,
    build_fidelity_circuit,
    circuit_to_bytes,
)
from DQuLearn.lib.cluster import LocalCluster, ManagerExecutor, WorkerSpec
from DQuLearn.lib.comanager import CoManager
from DQuLearn.lib.execution import CircuitTask, ExecutionSettings

EXACT = ExecutionSettings(shots=1, exact=True, compile_mode="cached")
LATENCY = 0.3


def _five_qubit_tasks(count: int) -> list[CircuitTask]:
    layout = QuClassiLayout(5, 1)
    rng = np.random.default_rng(0)
    circuit = build_fidelity_circuit(
        layout,
        rng.uniform(0, np.pi, len(layout.parameter_slots)),
        rng.uniform(0, np.pi, layout.n_features),
    )
    return [CircuitTask(circuit_to_bytes(circuit), 5, seed) for seed in range(count)]


@pytest.mark.parametrize("tenancy", ["multi", "single"])
def test_emulated_latency_overlaps_only_when_circuits_share_a_worker(tmp_path, tenancy):
    # One 20-qubit worker holds four 5-qubit circuits at once in multi-tenancy,
    # so their latencies overlap; in single-tenancy they run one after another.
    connections = 4 if tenancy == "multi" else 1
    with LocalCluster(
        [WorkerSpec(20, cpu_slots=1, latency_seconds=LATENCY)],
        EXACT,
        tmp_path,
        connections_per_worker=[connections],
    ) as cluster:
        with CoManager(
            cluster.handles,
            tenancy=tenancy,
            heartbeat_seconds=1.0,
            max_parallel_calls=connections,
        ) as manager:
            executor = ManagerExecutor(manager, "client")
            executor.run(_five_qubit_tasks(1))  # warm the worker's compiler
            start = time.perf_counter()
            executor.run(_five_qubit_tasks(4))
            elapsed = time.perf_counter() - start

    if tenancy == "multi":
        assert elapsed < 2.5 * LATENCY, elapsed
    else:
        assert elapsed > 3.8 * LATENCY, elapsed
