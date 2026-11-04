from __future__ import annotations

import numpy as np
import pytest
from common import (
    PROJECT_DIR,  # noqa: F401  (puts runtime_lib and the DQuLearn package on sys.path)
)
from DQuLearn.lib.circuits import QuClassiLayout, build_fidelity_circuit
from DQuLearn.lib.cluster import LocalCluster, ManagerExecutor, WorkerSpec
from DQuLearn.lib.comanager import CoManager
from DQuLearn.lib.data import SplitArrays
from DQuLearn.lib.execution import ExecutionSettings, LocalExecutor
from DQuLearn.lib.quclassi import (
    QuClassiModel,
    paper_circuits_per_epoch,
    resolve_circuit_budget,
)

EXACT = ExecutionSettings(shots=1, exact=True, compile_mode="per_circuit")
SHOTS = ExecutionSettings(shots=2000, exact=False, compile_mode="per_circuit")


def _split(n_samples: int, n_features: int, seed: int = 0) -> SplitArrays:
    rng = np.random.default_rng(seed)
    features = rng.uniform(0, 1, size=(n_samples, n_features))
    labels = np.arange(n_samples) % 2
    return SplitArrays(2 * np.arcsin(np.sqrt(features)), features, labels)


def _train(model, split, executor, **overrides):
    options = {
        "epoch_index": 0,
        "lr": 0.1,
        "update_rule": "simultaneous",
        "shift_schedule": "fixed",
        "dispatch_unit": "sample",
        "circuit_budget": None,
        "run_seed": 11,
    }
    options.update(overrides)
    return model.train_epoch(split, executor, **options)


@pytest.mark.parametrize(
    ("qubits", "layers", "circuits"),
    [(5, 1, 1440), (5, 3, 4320), (7, 1, 2016), (7, 3, 6048)],
)
def test_paper_circuit_counts(qubits, layers, circuits):
    assert paper_circuits_per_epoch(QuClassiLayout(qubits, layers)) == circuits


def test_circuit_budget_values():
    layout = QuClassiLayout(5, 2)
    assert resolve_circuit_budget("paper", layout) == 2880
    assert resolve_circuit_budget("all", layout) is None
    assert resolve_circuit_budget("80", layout) == 80
    with pytest.raises(ValueError):
        resolve_circuit_budget(7, layout)


def test_simultaneous_update_matches_a_hand_computation():
    layout = QuClassiLayout(5, 1)
    split = _split(1, layout.n_features)
    model = QuClassiModel(layout, 2, "pi", seed=1)
    start = model.parameters.copy()
    _train(model, split, LocalExecutor(EXACT), lr=0.1)

    def loss(parameters):
        circuit = build_fidelity_circuit(
            layout, parameters, split.angles[0], measure=False
        )
        from qiskit.quantum_info import Statevector

        return -np.log(max(2 * Statevector(circuit).probabilities([0])[0] - 1, 1e-16))

    expected = start.copy()
    for index in range(layout.n_parameters):
        plus, minus = start[0].copy(), start[0].copy()
        plus[index] += np.pi / 2
        minus[index] -= np.pi / 2
        expected[0, index] -= 0.1 * 0.5 * (loss(plus) - loss(minus))
    np.testing.assert_allclose(model.parameters, expected, atol=1e-12)


def test_sequential_update_sees_the_previous_parameter_update():
    layout = QuClassiLayout(5, 1)
    split = _split(1, layout.n_features)
    simultaneous = QuClassiModel(layout, 2, "pi", seed=1)
    sequential = QuClassiModel(layout, 2, "pi", seed=1)
    _train(simultaneous, split, LocalExecutor(EXACT))
    _train(sequential, split, LocalExecutor(EXACT), update_rule="sequential")
    # The first parameter sees the same state under both rules; the next ones do not.
    assert sequential.parameters[0, 0] == pytest.approx(simultaneous.parameters[0, 0])
    assert not np.allclose(sequential.parameters[0, 1:], simultaneous.parameters[0, 1:])


def test_budget_stops_inside_a_sample():
    layout = QuClassiLayout(5, 1)  # 4 parameters, 8 circuits per sample
    model = QuClassiModel(layout, 2, "pi", seed=1)
    stats = _train(
        model, _split(3, layout.n_features), LocalExecutor(EXACT), circuit_budget=12
    )
    assert (stats.circuits, stats.samples) == (12, 2)


def test_training_through_two_workers_is_identical_to_local_training(tmp_path):
    # Each circuit carries its own simulator seed, so with shots too the result
    # cannot depend on which worker ran a circuit: distribution changes nothing.
    layout = QuClassiLayout(5, 2)
    split = _split(4, layout.n_features)
    local_model = QuClassiModel(layout, 2, "pi", seed=2)
    _train(local_model, split, LocalExecutor(SHOTS))
    local_accuracy = local_model.evaluate(
        split, LocalExecutor(SHOTS), run_seed=11, evaluation_key=1
    )

    distributed_model = QuClassiModel(layout, 2, "pi", seed=2)
    with LocalCluster(
        [WorkerSpec(5), WorkerSpec(5)], SHOTS, tmp_path, connections_per_worker=[1, 1]
    ) as cluster:
        with CoManager(cluster.handles, heartbeat_seconds=1.0) as manager:
            executor = ManagerExecutor(manager, "client")
            _train(distributed_model, split, executor)
            distributed_accuracy = distributed_model.evaluate(
                split, executor, run_seed=11, evaluation_key=1
            )
            used = {
                state["worker"]: state["assigned_total"] for state in manager.snapshot()
            }

    assert all(count > 0 for count in used.values()), used
    np.testing.assert_array_equal(distributed_model.parameters, local_model.parameters)
    assert distributed_accuracy == local_accuracy
