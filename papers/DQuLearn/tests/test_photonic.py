"""MerLin adaptation: the photonic circuit, its gradient rules and its execution path."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from common import (
    PROJECT_DIR,  # noqa: F401  (puts runtime_lib and the DQuLearn package on sys.path)
)
from DQuLearn.lib.cluster import LocalCluster, ManagerExecutor, WorkerSpec
from DQuLearn.lib.comanager import CoManager
from DQuLearn.lib.data import SplitArrays
from DQuLearn.lib.execution import CircuitRunner, ExecutionSettings, LocalExecutor
from DQuLearn.lib.photonic_merlin import (
    PhotonicFidelityRunner,
    PhotonicLayout,
    build_photonic_circuit,
    decode_task,
    encode_task,
    exact_shift_rule,
)
from DQuLearn.lib.quclassi import PhotonicQuClassiModel

MERLIN = ExecutionSettings(
    shots=1, exact=True, compile_mode="per_circuit", workload="merlin"
)
LAYOUT = PhotonicLayout(5)  # stands for QuClassi's 5-qubit circuit: 2 photons, 4 modes


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


def _circuit_unitary(layout, parameters, angles):
    circuit = build_photonic_circuit(layout)
    for parameter in circuit.get_parameters():
        name = parameter.name
        value = (
            angles[int(name[1:])]
            if name.startswith("x")
            else parameters[int(name.split("_")[1])]
        )
        parameter.set_value(float(value))
    return np.array(circuit.compute_unitary())


def _permanent_probability(layout, parameters, angles):
    """Probability of the input state from the circuit unitary (2 photons: 2x2 permanent)."""
    unitary = _circuit_unitary(layout, parameters, angles)
    occupied = [mode for mode, n in enumerate(layout.input_state) if n]
    block = unitary[np.ix_(occupied, occupied)]
    return abs(block[0, 0] * block[1, 1] + block[0, 1] * block[1, 0]) ** 2


def test_layout_counts():
    assert (LAYOUT.n_photons, LAYOUT.n_modes, LAYOUT.n_features) == (2, 4, 4)
    assert LAYOUT.input_state == (1, 0, 1, 0)
    assert LAYOUT.demand == 4 and LAYOUT.max_frequency == 2
    names = [p.name for p in build_photonic_circuit(LAYOUT).get_parameters()]
    assert names == [f"x{i}" for i in range(4)] + [f"theta_{i:03d}" for i in range(12)]
    assert LAYOUT.n_parameters == 12
    with pytest.raises(ValueError):
        PhotonicLayout(5, n_layers=2)


def test_task_round_trip():
    rng = np.random.default_rng(0)
    parameters, angles = rng.uniform(0, 6, 12), rng.uniform(0, 3, 4)
    layout, decoded_parameters, decoded_angles = decode_task(
        encode_task(LAYOUT, parameters, angles)
    )
    assert layout == LAYOUT
    np.testing.assert_array_equal(decoded_parameters, parameters)
    np.testing.assert_array_equal(decoded_angles, angles)


def test_merlin_probability_equals_an_independent_permanent():
    # Checks MerLin's parameter order, the FOCK space and the read-out index.
    runner = PhotonicFidelityRunner()
    rng = np.random.default_rng(1)
    for _ in range(5):
        parameters, angles = rng.uniform(0, 2 * np.pi, 12), rng.uniform(0, np.pi, 4)
        assert runner.probability(LAYOUT, parameters, angles) == pytest.approx(
            _permanent_probability(LAYOUT, parameters, angles), abs=1e-12
        )


def test_data_states_have_the_qubit_overlaps():
    # Dual rail: two data points overlap as QuClassi's RY-then-RZ qubit states do.
    def qubit(a, b):
        return np.array([np.cos(a / 2), np.exp(1j * b) * np.sin(a / 2)])

    def photon_states(angles):
        # The encoding acts on each pair of modes separately: photon p's state is
        # column 2p of the encoding unitary, restricted to modes 2p and 2p + 1.
        # The mesh is the identity when its circuit is left out, so build the
        # encoding alone with the same components as build_photonic_circuit.
        import perceval as pcvl

        encoding = pcvl.Circuit(4)
        for photon in range(2):
            mode = 2 * photon
            encoding.add(mode, pcvl.BS()).add(mode, pcvl.PS(float(angles[2 * photon])))
            encoding.add(mode, pcvl.BS())
            encoding.add(mode, pcvl.PS(float(angles[2 * photon + 1])))
        unitary = np.array(encoding.compute_unitary())
        return [unitary[2 * p : 2 * p + 2, 2 * p] for p in range(2)]

    rng = np.random.default_rng(2)
    for _ in range(50):
        u, v = rng.uniform(0, np.pi, 4), rng.uniform(0, np.pi, 4)
        qubit_overlap = np.prod(
            [
                abs(
                    np.vdot(
                        qubit(u[2 * p], u[2 * p + 1]), qubit(v[2 * p], v[2 * p + 1])
                    )
                )
                ** 2
                for p in range(2)
            ]
        )
        photon_overlap = np.prod(
            [
                abs(np.vdot(a, b)) ** 2
                for a, b in zip(photon_states(u), photon_states(v))
            ]
        )
        assert photon_overlap == pytest.approx(qubit_overlap, abs=1e-12)


def _autograd_loss_gradient(parameters, angles):
    """d(−log F)/dθ by autograd through MerLin."""
    runner = PhotonicFidelityRunner()
    layer, phases, index = runner._layer(LAYOUT)
    with torch.no_grad():
        phases.copy_(torch.as_tensor(parameters))
    loss = -torch.log(layer(torch.as_tensor(angles)[None])[0, index])
    (gradient,) = torch.autograd.grad(loss, [phases])
    return gradient.numpy()


def test_exact_shift_rule_values():
    shifts, weights = exact_shift_rule(1)
    np.testing.assert_allclose(shifts, [np.pi / 2, 3 * np.pi / 2])
    np.testing.assert_allclose(weights, [0.5, -0.5])
    shifts, _ = exact_shift_rule(2)
    np.testing.assert_allclose(shifts, np.pi / 4 * np.array([1, 3, 5, 7]))


def test_exact_shift_update_equals_autograd_and_two_point_does_not():
    split = _split(1, LAYOUT.n_features, seed=3)
    angles, lr = split.angles[0], 0.1
    exact = PhotonicQuClassiModel(LAYOUT, 2, "pi", seed=4)
    two_point = PhotonicQuClassiModel(LAYOUT, 2, "pi", seed=4)
    start = exact.parameters[0].copy()
    expected = start - lr * _autograd_loss_gradient(start, angles)

    stats = _train(
        exact, split, LocalExecutor(MERLIN), lr=lr, gradient_rule="exact_shift"
    )
    _train(two_point, split, LocalExecutor(MERLIN), lr=lr)

    assert stats.circuits == 1 + 4 * 12 == exact.circuits_per_sample("exact_shift")
    np.testing.assert_allclose(exact.parameters[0], expected, atol=1e-10)
    assert np.max(np.abs(two_point.parameters[0] - expected)) > 1e-3
    np.testing.assert_array_equal(exact.parameters[1], two_point.parameters[1])


def test_exact_shift_rule_is_refused_where_it_does_not_apply():
    model = PhotonicQuClassiModel(LAYOUT, 2, "pi", seed=0)
    split = _split(2, LAYOUT.n_features)
    with pytest.raises(ValueError):
        _train(
            model,
            split,
            LocalExecutor(MERLIN),
            gradient_rule="exact_shift",
            update_rule="sequential",
        )
    with pytest.raises(ValueError):
        _train(
            model,
            split,
            LocalExecutor(MERLIN),
            gradient_rule="exact_shift",
            circuit_budget=10,
        )


def test_merlin_workload_refuses_shots():
    with pytest.raises(ValueError):
        ExecutionSettings(
            shots=100, exact=False, compile_mode="per_circuit", workload="merlin"
        )


def test_circuit_runner_dispatches_to_merlin_and_floors_the_fidelity():
    runner = CircuitRunner(MERLIN)
    rng = np.random.default_rng(5)
    parameters, angles = rng.uniform(0, 2 * np.pi, 12), rng.uniform(0, np.pi, 4)
    assert runner.run(encode_task(LAYOUT, parameters, angles), 0) == pytest.approx(
        _permanent_probability(LAYOUT, parameters, angles), abs=1e-12
    )
    # A zero probability is floored at 1e-16, as QuClassi's code does, so −log F
    # stays finite.
    photonic = runner._photonic
    photonic.probability = lambda *_: 0.0
    zero = runner.run(encode_task(LAYOUT, parameters, angles), 0)
    assert zero == 1e-16
    assert math.isfinite(-math.log(zero))


def test_photonic_training_through_two_workers_is_identical_to_local(tmp_path):
    split = _split(4, LAYOUT.n_features)
    local_model = PhotonicQuClassiModel(LAYOUT, 2, "pi", seed=2)
    _train(local_model, split, LocalExecutor(MERLIN))
    local_accuracy = local_model.evaluate(
        split, LocalExecutor(MERLIN), run_seed=11, evaluation_key=1
    )

    distributed_model = PhotonicQuClassiModel(LAYOUT, 2, "pi", seed=2)
    with LocalCluster(
        [WorkerSpec(4), WorkerSpec(4)], MERLIN, tmp_path, connections_per_worker=[1, 1]
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
