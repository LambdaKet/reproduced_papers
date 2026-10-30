"""Summary metrics with the rules predeclared in the LOG (phase 01).

- Final fidelity of a seed: mean over the last ``window`` rounds.
- Convergence round of a seed: first round after which the trailing
  ``window``-round mean stays within ``tolerance`` of the final fidelity until
  the end of the run. The paper does not define its "convergence round"; ours
  is compared with it as a trend only.
- Round reaching a threshold: first round at which the mean fidelity over
  seeds reaches it (None if never).
"""

from __future__ import annotations

import numpy as np


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over rounds 1..R of ``values`` ``(R + 1, S)``.

    Row r of the result (r >= 1) averages rounds max(1, r - window + 1)..r;
    row 0 is round 0 itself.
    """
    rounds = values.shape[0] - 1
    cumulative = np.cumsum(values[1:], axis=0)
    out = np.empty_like(values)
    out[0] = values[0]
    for r in range(1, rounds + 1):
        start = max(1, r - window + 1)
        total = cumulative[r - 1] - (cumulative[start - 2] if start > 1 else 0.0)
        out[r] = total / (r - start + 1)
    return out


def final_fidelity(fidelity: np.ndarray, window: int) -> np.ndarray:
    """Per-seed mean over the last ``window`` rounds of ``fidelity`` ``(R + 1, S)``."""
    if window < 1 or window > fidelity.shape[0] - 1:
        raise ValueError(
            f"window must be in [1, {fidelity.shape[0] - 1}], got {window}"
        )
    return fidelity[-window:].mean(axis=0)


def convergence_round(
    fidelity: np.ndarray, window: int, tolerance: float
) -> np.ndarray:
    """Per-seed convergence round (int array), see the module docstring."""
    final = final_fidelity(fidelity, window)
    trailing = trailing_mean(fidelity, window)
    outside = np.abs(trailing[1:] - final) > tolerance
    rounds = fidelity.shape[0] - 1
    result = np.empty(fidelity.shape[1], dtype=int)
    for seed_index in range(fidelity.shape[1]):
        hits = np.nonzero(outside[:, seed_index])[0]
        result[seed_index] = 1 if hits.size == 0 else min(hits[-1] + 2, rounds)
    return result


def first_round_reaching(mean_curve: np.ndarray, threshold: float) -> int | None:
    """First round (index of ``mean_curve``) at or above ``threshold``, else None."""
    hits = np.nonzero(mean_curve >= threshold)[0]
    return None if hits.size == 0 else int(hits[0])


def summarize(values: np.ndarray) -> dict[str, float]:
    """Mean, standard deviation (ddof=1 when possible), min and max of a 1-D array."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
    }


def shot_noise_tvd(
    count_levels: list[int], repeats: int, rng: np.random.Generator
) -> dict[int, float]:
    """Mean total variation distance of a sampled uniform 16-outcome distribution.

    Suppl. Sec. 5.2: each outcome count is Poisson with mean N / 16; the
    estimate is the normalized counts; TVD = 1/2 sum |P_the - P_exp|.

    Parameters
    ----------
    count_levels : list of int
        Mean total counts N.
    repeats : int
        Repeats per level (1000 in the paper).
    rng : numpy.random.Generator
        Random stream.

    Returns
    -------
    dict
        N -> mean TVD.
    """
    uniform = np.full(16, 1.0 / 16.0)
    out = {}
    for level in count_levels:
        counts = rng.poisson(level * uniform, size=(repeats, 16))
        totals = counts.sum(axis=1, keepdims=True)
        estimate = np.divide(
            counts, totals, out=np.zeros(counts.shape), where=totals > 0
        )
        out[int(level)] = float(0.5 * np.abs(estimate - uniform).sum(axis=1).mean())
    return out
