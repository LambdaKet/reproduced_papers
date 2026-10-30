"""The 7 tests of the Phase 2 data-generation specification (LOG, Data Acquisition Log)."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import pytest
from qgan_quantum_data_photonics.lib.data import (
    draw_target,
    generate_inputs,
    seed_stream,
)
from qgan_quantum_data_photonics.lib.metrics import shot_noise_tvd
from qgan_quantum_data_photonics.lib.qgan import QganSettings, _Hardware, train_qgan


def test_target_shape_dtype_and_norm():
    tau = draw_target(0)
    assert tau.shape == (16,)
    assert np.iscomplexobj(tau)
    assert abs(np.linalg.norm(tau) - 1.0) < 1e-12


def test_targets_are_maximally_entangled():
    for seed in range(50):
        amplitudes = draw_target(seed).reshape(4, 4)
        reduced = amplitudes @ amplitudes.conj().T
        assert np.abs(reduced - np.eye(4) / 4).max() < 1e-12
        schmidt = np.linalg.svd(amplitudes, compute_uv=False)
        assert np.abs(schmidt - 0.5).max() < 1e-12


def test_targets_follow_the_haar_measure():
    traces = np.array(
        [np.trace(2.0 * draw_target(s).reshape(4, 4)) for s in range(20000)]
    )
    assert abs(np.mean(np.abs(traces) ** 2) - 1.0) < 0.03
    identity_fidelity = np.abs(traces) / 4.0
    assert abs(identity_fidelity.mean() - 0.22) < 0.01


def test_inputs_are_deterministic_and_paired():
    first = generate_inputs(
        [3, 4], {"mode": "random_per_circuit", "count_per_circuit": 5}
    )
    again = generate_inputs(
        [3, 4], {"mode": "random_per_circuit", "count_per_circuit": 5}
    )
    alone = generate_inputs([4], {"mode": "random_per_circuit", "count_per_circuit": 5})
    for name in ("targets", "discriminator_init", "stuck_mask", "stuck_values"):
        assert np.array_equal(getattr(first, name), getattr(again, name))
        assert np.array_equal(getattr(first, name)[1], getattr(alone, name)[0])
    intact = generate_inputs([3, 4], {"mode": "none"})
    assert np.array_equal(first.targets, intact.targets)
    assert np.array_equal(first.discriminator_init, intact.discriminator_init)
    assert not np.allclose(first.targets[0], first.targets[1])


@pytest.mark.parametrize("count", [0, 1, 8, 12])
def test_defects_count_values_and_immobility(count):
    inputs = generate_inputs(
        [0, 1, 2], {"mode": "random_per_circuit", "count_per_circuit": count}
    )
    assert (inputs.stuck_mask.sum(axis=2) == count).all()
    assert ((inputs.stuck_values >= 0) & (inputs.stuck_values < 2 * np.pi)).all()
    result = train_qgan(
        inputs,
        QganSettings(
            rounds=3,
            steps_per_turn=2,
            learning_rate=0.3 * np.pi,
            shift=np.pi / 4,
            phase_sigma=0.02 * np.pi,
            counts=1500,
            eval_every=3,
            noise_draws=2,
        ),
    )
    stuck = inputs.stuck_mask
    assert np.array_equal(result.generator_phases[stuck], inputs.stuck_values[stuck])


def test_phase_noise_statistics():
    inputs = generate_inputs([0], {"mode": "none"})
    quiet = _Hardware(
        inputs,
        QganSettings(1, 1, 0.1, np.pi / 4, phase_sigma=0.0, counts=None),
    )
    assert quiet._noise(np.arange(1), (10, 30), 0) is None
    sigma = 0.03 * np.pi
    noisy = _Hardware(
        inputs, QganSettings(1, 1, 0.1, np.pi / 4, phase_sigma=sigma, counts=None)
    )
    draws = noisy._noise(np.arange(1), (100000 // 30 + 1, 30), 0)
    assert abs(draws.std() / sigma - 1.0) < 0.02


def test_count_model_matches_suppl_fig_9a():
    rng = seed_stream(0, "counts")
    totals = rng.poisson(1500 / 16, size=(1000, 16)).sum(axis=1)
    assert abs(totals.mean() / 1500 - 1.0) < 0.01
    tvd = shot_noise_tvd([100, 1500, 10000], 1000, seed_stream(0, "counts"))
    assert abs(tvd[100] - 0.154) < 0.005
    assert abs(tvd[1500] - 0.040) < 0.005
    assert abs(tvd[10000] - 0.015) < 0.005
