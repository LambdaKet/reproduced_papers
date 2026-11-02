"""MerLin adaptation (phase 04): the chip as a MerLin layer inside the same game."""

from __future__ import annotations

import json
from importlib.metadata import version

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
import perceval as pcvl
import pytest
import torch
from common import load_runtime_ready_config
from qgan_quantum_data_photonics.lib.circuits import (
    beam_splitter,
    generated_amplitudes,
    output_distribution,
    root_fidelity,
)
from qgan_quantum_data_photonics.lib.data import generate_inputs
from qgan_quantum_data_photonics.lib.qgan import QganSettings, _Hardware, train_qgan
from qgan_quantum_data_photonics.lib.qgan_merlin import (
    MerlinHardware,
    build_chip_layer,
    chip_probabilities,
    coincidence_index,
    target_preparation_phases,
)

RNG = np.random.default_rng(5)


def _settings(**overrides):
    base = {
        "rounds": 4,
        "steps_per_turn": 20,
        "learning_rate": 0.3 * np.pi,
        "shift": np.pi / 4,
        "phase_sigma": 0.02 * np.pi,
        "counts": 1500,
        "generator_stop_d": 0.05,
        "eval_every": 4,
        "noise_draws": 5,
    }
    base.update(overrides)
    return QganSettings(**base)


def test_perceval_beam_splitter_is_the_chip_mmi():
    assert np.abs(np.array(pcvl.BS().compute_unitary()) - beam_splitter()).max() < 1e-12


def test_layer_space_detector_and_outputs():
    layer = build_chip_layer()
    assert layer.output_size == 28  # C(8, 2): UNBUNCHED, threshold detection, 2 photons
    index = coincidence_index(layer)
    assert sorted(index.ravel().tolist()) == sorted(set(index.ravel().tolist()))
    assert sum(p.numel() for p in layer.parameters()) == 0


def test_chip_layer_matches_the_numpy_simulator():
    layer = build_chip_layer()
    index = coincidence_index(layer)
    phases = RNG.uniform(0, 2 * np.pi, (40, 42))
    merlin = chip_probabilities(layer, phases, index)
    reference = output_distribution(
        generated_amplitudes(phases[:, :15], phases[:, 15:30]),
        phases[:, 30:36],
        phases[:, 36:],
    )
    assert np.abs(merlin - reference).max() < 1e-12
    assert np.abs(merlin.sum(axis=(1, 2)) - 1.0).max() < 1e-12  # nothing outside the 16


def test_targets_are_prepared_on_the_chip():
    inputs = generate_inputs([0, 1, 2], {"mode": "none"})
    phases = target_preparation_phases(inputs)
    prepared = generated_amplitudes(phases[:, :15], phases[:, 15:])
    assert (root_fidelity(prepared, inputs.target_amplitudes) > 1 - 1e-12).all()


def test_injected_mode_reproduces_the_numpy_game_exactly():
    inputs = generate_inputs(
        [0, 1, 2], {"mode": "random_per_circuit", "count_per_circuit": 3}
    )
    settings = _settings()
    reference = train_qgan(inputs, settings)
    merlin = train_qgan(inputs, settings, MerlinHardware(inputs, settings, "injected"))
    assert np.abs(reference.fidelity - merlin.fidelity).max() < 1e-12
    assert np.abs(reference.d_measured - merlin.d_measured).max() < 1e-12


def test_phase_error_mode_uses_merlin_noise_only():
    inputs = generate_inputs([0], {"mode": "none"})
    sigma = 0.03 * np.pi
    hardware = MerlinHardware(inputs, _settings(phase_sigma=sigma), "phase_error")
    assert hardware.phase_error == pytest.approx(np.sqrt(3.0) * sigma)
    assert (
        hardware._noise(np.arange(1), (5, 30), 0) is None
    )  # nothing injected by lib.qgan
    phases = np.repeat(RNG.uniform(0, 2 * np.pi, (1, 42)), 4, axis=0)
    noisy = chip_probabilities(hardware.layer, phases, hardware.index)
    exact = chip_probabilities(hardware.exact_layer, phases, hardware.index)
    assert not np.allclose(noisy[0], noisy[1])
    assert np.allclose(exact[0], exact[1])


def test_phase_error_mode_is_deterministic_under_a_torch_seed():
    inputs = generate_inputs([0, 1], {"mode": "none"})
    settings = _settings(rounds=2)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        runs.append(
            train_qgan(
                inputs, settings, MerlinHardware(inputs, settings, "phase_error")
            )
        )
    assert np.array_equal(runs[0].fidelity, runs[1].fidelity)


def test_merlin_sampler_matches_the_count_model():
    """MerLin's own sampler (multinomial, 1500 shots) against our Poisson counts."""
    layer = build_chip_layer()
    index = coincidence_index(layer)
    phases = np.repeat(RNG.uniform(0, 2 * np.pi, (1, 42)), 300, axis=0)
    columns = [torch.from_numpy(phases[:, j : j + 1].copy()) for j in range(42)]
    torch.manual_seed(1)
    with torch.no_grad():
        sampled = layer(*columns, shots=1500).numpy()[:, index][:, 1, 1]
    exact = chip_probabilities(layer, phases[:1], index)
    hardware = _Hardware(
        generate_inputs([0], {"mode": "none"}), _settings(phase_sigma=0.0)
    )
    ours = hardware._estimate(
        np.zeros(1, dtype=int), np.repeat(exact[None], 300, axis=1)
    )[0]
    p = exact[0, 1, 1]
    standard_error = np.sqrt(p * (1 - p) / 1500 / 300)
    assert abs(sampled.mean() - p) < 4 * standard_error
    assert abs(ours.mean() - p) < 4 * standard_error
    assert 0.7 < sampled.var() / ours.var() < 1.4


def test_runner_routes_the_merlin_backend(tmp_path):
    from qgan_quantum_data_photonics.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["backend"] = {"name": "merlin", "merlin_noise": "injected"}
    cfg["training"]["rounds"] = 3
    cfg["evaluation"]["final_window"] = 3
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, default=str))
    paper_runner.train_and_evaluate(cfg, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["backend"]["name"] == "merlin"
    assert metrics["hardware"]["computation_space"] == "UNBUNCHED"
    assert metrics["hardware"]["photon_number"] == 2
    assert (
        f"merlinquantum {version('merlinquantum')}" in metrics["hardware"]["framework"]
    )
