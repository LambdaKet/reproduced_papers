"""Gradients, count estimator, seeding and learning of the QGAN game."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
from qgan_quantum_data_photonics.lib.ceiling import oracle_ceiling
from qgan_quantum_data_photonics.lib.circuits import (
    generated_amplitudes,
    output_distribution,
)
from qgan_quantum_data_photonics.lib.data import generate_inputs
from qgan_quantum_data_photonics.lib.qgan import (
    QganSettings,
    _Hardware,
    _shift_gradient,
    _shifted,
    train_qgan,
)

RNG = np.random.default_rng(11)


def _generated_p22(g, d):
    amplitudes = generated_amplitudes(g[:, :15], g[:, 15:])
    return output_distribution(amplitudes, d[:, :6], d[:, 6:])[:, 1, 1]


def test_parameter_shift_is_exact():
    """Each phase shifter carries at most one photon: the pi/4 rule is exact."""
    g, d = RNG.uniform(0, 2 * np.pi, (1, 30)), RNG.uniform(0, 2 * np.pi, (1, 12))
    shift, eps = np.pi / 4, 1e-6
    shifted_g = _shifted(g, shift)[0]
    values = _generated_p22(shifted_g, np.repeat(d, len(shifted_g), axis=0))[None]
    rule = _shift_gradient(values, shift)[0]
    finite = [
        (_generated_p22(g + eps * e, d) - _generated_p22(g - eps * e, d))[0] / (2 * eps)
        for e in np.eye(30)[:, None, :]
    ]
    assert np.abs(rule - np.array(finite)).max() < 1e-8
    shifted_d = _shifted(d, shift)[0]
    values = _generated_p22(np.repeat(g, len(shifted_d), axis=0), shifted_d)[None]
    rule = _shift_gradient(values, shift)[0]
    finite = [
        (_generated_p22(g, d + eps * e) - _generated_p22(g, d - eps * e))[0] / (2 * eps)
        for e in np.eye(12)[:, None, :]
    ]
    assert np.abs(rule - np.array(finite)).max() < 1e-8


def test_count_estimator_is_unbiased():
    inputs = generate_inputs([0], {"mode": "none"})
    hardware = _Hardware(
        inputs, QganSettings(1, 1, 0.1, np.pi / 4, phase_sigma=0.0, counts=1500)
    )
    g = np.repeat(RNG.uniform(0, 2 * np.pi, (1, 1, 30)), 4000, axis=1)
    d = np.repeat(RNG.uniform(0, 2 * np.pi, (1, 1, 12)), 4000, axis=1)
    exact = _generated_p22(g[0, :1], d[0, :1])[0]
    estimates = hardware.measure_generated(g, d)[0]
    standard_error = np.sqrt(exact * (1 - exact) / 1500 / 4000)
    assert abs(estimates.mean() - exact) < 5 * standard_error + 1e-9


def _settings(**overrides):
    base = {
        "rounds": 60,
        "steps_per_turn": 20,
        "learning_rate": 0.3 * np.pi,
        "shift": np.pi / 4,
        "phase_sigma": 0.01 * np.pi,
        "counts": 1500,
        "generator_stop_d": 0.05,
        "eval_every": 60,
        "noise_draws": 10,
    }
    base.update(overrides)
    return QganSettings(**base)


def test_seed_results_do_not_depend_on_the_batch():
    settings = _settings(rounds=5)
    both = train_qgan(generate_inputs([5, 6], {"mode": "none"}), settings)
    alone = train_qgan(generate_inputs([6], {"mode": "none"}), settings)
    assert np.array_equal(both.fidelity[:, 1], alone.fidelity[:, 0])


def test_noise_free_qgan_learns_the_target():
    inputs = generate_inputs([2000, 2001, 2002, 2003], {"mode": "none"})
    result = train_qgan(inputs, _settings(phase_sigma=0.0, counts=None, rounds=150))
    assert result.fidelity[0].mean() < 0.5
    assert result.fidelity[-1].min() > 0.9


def test_ceiling_reaches_one_on_intact_circuits():
    inputs = generate_inputs([0, 1], {"mode": "none"})
    best, phases = oracle_ceiling(
        inputs, restarts=3, iterations=400, learning_rate=0.05, device="cpu"
    )
    assert best.min() > 0.999
    assert phases.shape == (2, 2, 15)
