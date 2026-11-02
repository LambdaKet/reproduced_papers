"""Phase-05 baselines: matched with the reproduced experiments they are compared with."""

from __future__ import annotations

import json

from common import PROJECT_DIR

CONFIGS = PROJECT_DIR / "configs"


def _overlay(name: str) -> dict:
    return json.loads((CONFIGS / f"{name}.json").read_text())


def test_noise_free_reference_changes_only_the_noise():
    reference = _overlay("noise_free_reference_original")
    phase_noise = _overlay("phase_noise_sweep_original")
    assert reference["noise"] == {"phase_sigma_pi": 0.0, "counts": None}
    for config in (reference, phase_noise):
        for key in ("description", "experiment", "noise"):
            config.pop(key)
    assert reference == phase_noise


def test_noise_free_reference_spec_matches_the_run_lengths():
    spec = json.loads(
        (
            PROJECT_DIR / "utils" / "sweep_specs" / "noise_free_reference.json"
        ).read_text()
    )
    rounds = {c["name"]: c["args"] for c in spec["candidates"]}
    assert rounds == {
        "noise_free_400_rounds": [],  # Fig. 2c (configs/phase_noise_sweep_original.json)
        "noise_free_2500_rounds": ["--rounds", "2500"],  # Suppl. Fig. 9b
    }
    assert _overlay("shot_noise_sweep_original")["training"]["rounds"] == 2500
