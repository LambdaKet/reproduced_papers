"""Predeclared summary rules: final fidelity, convergence round, first round reaching."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)
import numpy as np
from qgan_quantum_data_photonics.lib.metrics import (
    convergence_round,
    final_fidelity,
    first_round_reaching,
    trailing_mean,
)


def _curve():
    # Rounds 0..100: rises linearly to 0.9 at round 50, then stays.
    rounds = np.arange(101)
    return np.minimum(0.9, 0.018 * rounds)[:, None]


def test_final_fidelity_is_the_mean_of_the_last_rounds():
    curve = np.arange(11, dtype=float)[:, None]
    assert final_fidelity(curve, 4)[0] == np.mean([7, 8, 9, 10])


def test_trailing_mean():
    curve = np.arange(6, dtype=float)[:, None]
    assert np.allclose(trailing_mean(curve, 2)[:, 0], [0, 1, 1.5, 2.5, 3.5, 4.5])


def test_convergence_round_of_a_plateau():
    # At round r the 20-round window still holds m = 69 - r rising values, whose
    # shortfall from 0.9 sums to 0.018 m (m + 1) / 2. The trailing mean is within
    # 0.01 of 0.9 once that sum is at most 0.2: m <= 4, so from round 65 on.
    assert convergence_round(_curve(), 20, 0.01)[0] == 65


def test_first_round_reaching():
    assert first_round_reaching(_curve()[:, 0], 0.89) == 50
    assert first_round_reaching(_curve()[:, 0], 0.95) is None
