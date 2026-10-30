"""The paper's QGAN game on simulated hardware, batched over seeds.

One round (main text): the discriminator takes ``steps_per_turn`` gradient
steps that maximize d = |tr[M rho] - tr[M tau]| (Eq. 1), then the generator
takes ``steps_per_turn`` steps that minimize it. Gradients use the
parameter-shift rule of the cited method paper (arXiv:2310.07371),
df/dtheta = [f(theta + s) - f(theta - s)] / (2 sin s) with s = pi/4. Each
phase shifter carries at most one photon, so the rule is exact on noiseless
probabilities; on hardware every shifted term is a separate noisy execution.

Hardware model of one execution: every generator and discriminator phase gets
Gaussian noise N(0, sigma^2) (Fig. 1(e)); the 16 coincidence counts are Poisson
with total mean ``counts``, and tr[M rho] is estimated as c_22 / sum(c)
(Suppl. Sec. 5.2). Stuck generator phase shifters keep their fixed value and
are never updated (Fig. 1(f)); they still get phase noise. All random draws
come from per-seed streams, so a seed's run does not depend on the other seeds
of the batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .circuits import (
    DETECTION_PORT,
    DISCRIMINATOR_NUM_PHASES,
    GENERATOR_NUM_PHASES,
    generated_amplitudes,
    generator_identity_phases,
    output_distribution,
    root_fidelity,
)
from .data import SeedInputs, seed_stream

logger = logging.getLogger(__name__)

G_SIZE = 2 * GENERATOR_NUM_PHASES  # signal then idler
D_SIZE = 2 * DISCRIMINATOR_NUM_PHASES


@dataclass(frozen=True)
class QganSettings:
    """Training and hardware settings of one QGAN run.

    Attributes
    ----------
    rounds : int
        Number of rounds.
    steps_per_turn : int
        Gradient steps of each player per round.
    learning_rate : float
        Gradient step size in radians, the same for both players.
    shift : float
        Parameter-shift offset in radians (pi/4 in the cited method paper).
    phase_sigma : float
        Standard deviation of the phase noise in radians.
    counts : int or None
        Mean total coincidences per measurement setting; None gives exact
        probabilities.
    noise_redraw : str
        ``execution``: new phase noise at every execution (our default);
        ``round``: one draw per round, shared by all executions of the round.
    tau_measurement : str
        ``noisy``: tr[M tau] is measured with the same discriminator noise and
        counts as tr[M rho] (our default); ``exact``: computed from the known
        tau with the intended discriminator phases.
    discriminator : str
        ``adversarial`` (the paper) or ``random_fixed`` (our ablation: random
        projectors redrawn each round, no discriminator updates).
    generator_stop_d : float or None
        If set, the generator ends its turn as soon as the measured d is at or
        below this value, so ``steps_per_turn`` is its maximum; the
        discriminator always takes ``steps_per_turn`` steps. The paper does not
        state the rule; in Fig. 2a the generator's turns last 2-6 steps and end
        with a gap of about 0.02-0.04. Default value is None.
    eval_every : int
        Rounds between evaluations of the noise-averaged fidelity.
    noise_draws : int
        Phase-noise draws per noise-averaged fidelity.
    """

    rounds: int
    steps_per_turn: int
    learning_rate: float
    shift: float
    phase_sigma: float
    counts: int | None
    noise_redraw: str = "execution"
    tau_measurement: str = "noisy"
    discriminator: str = "adversarial"
    generator_stop_d: float | None = None
    eval_every: int = 50
    noise_draws: int = 100

    def __post_init__(self) -> None:
        if self.rounds < 1 or self.steps_per_turn < 1:
            raise ValueError("rounds and steps_per_turn must be at least 1")
        if self.noise_redraw not in ("execution", "round"):
            raise ValueError(f"Unknown noise_redraw {self.noise_redraw!r}")
        if self.tau_measurement not in ("noisy", "exact"):
            raise ValueError(f"Unknown tau_measurement {self.tau_measurement!r}")
        if self.discriminator not in ("adversarial", "random_fixed"):
            raise ValueError(f"Unknown discriminator {self.discriminator!r}")
        if self.counts is not None and self.counts < 1:
            raise ValueError("counts must be a positive integer or None")


@dataclass
class QganResult:
    """Trajectories of a QGAN run, seeds along the last axis.

    Attributes
    ----------
    fidelity : numpy.ndarray
        Shape ``(rounds + 1, S)``: root fidelity of the state prepared with the
        intended generator phases, before round 1 and after each round.
    d_measured : numpy.ndarray
        Shape ``(rounds, S)``: the last d measured in the generator's turn
        (what the hardware would record).
    d_exact : numpy.ndarray
        Shape ``(rounds, S)``: d with the intended phases and exact
        probabilities, at the end of each round.
    noise_averaged : dict
        Round -> shape ``(S,)``: fidelity averaged over phase-noise draws.
    generator_phases, discriminator_phases : numpy.ndarray
        Final phases, shapes ``(S, 2, 15)`` and ``(S, 2, 6)``.
    """

    fidelity: np.ndarray
    d_measured: np.ndarray
    d_exact: np.ndarray
    noise_averaged: dict[int, np.ndarray] = field(default_factory=dict)
    generator_phases: np.ndarray | None = None
    discriminator_phases: np.ndarray | None = None


class _Hardware:
    """Noisy executions of the chip for a batch of seeds."""

    def __init__(self, inputs: SeedInputs, settings: QganSettings) -> None:
        self.settings = settings
        self.targets = inputs.target_amplitudes
        seeds = inputs.seeds.tolist()
        self.noise_rngs = [seed_stream(s, "phase_noise") for s in seeds]
        self.count_rngs = [seed_stream(s, "counts") for s in seeds]
        self.round_offsets: tuple[np.ndarray, np.ndarray] | None = None

    def start_round(self) -> None:
        """Draw the round's shared phase noise when ``noise_redraw == 'round'``."""
        if self.settings.noise_redraw != "round" or self.settings.phase_sigma == 0:
            self.round_offsets = None
            return
        sigma = self.settings.phase_sigma
        g = np.stack([rng.normal(0.0, sigma, G_SIZE) for rng in self.noise_rngs])
        d = np.stack([rng.normal(0.0, sigma, D_SIZE) for rng in self.noise_rngs])
        self.round_offsets = (g, d)

    def _rows(self, index: np.ndarray | None) -> np.ndarray:
        """Seed rows measured by a call; ``None`` means every seed."""
        return np.arange(len(self.noise_rngs)) if index is None else index

    def _noise(
        self, rows: np.ndarray, shape_tail: tuple[int, ...], which: int
    ) -> np.ndarray | None:
        """Phase noise ``(len(rows), E, n)``; ``which`` 0 = generator, 1 = discriminator.

        Only the measured seeds draw, so a seed's stream never depends on the
        other seeds of the batch.
        """
        sigma = self.settings.phase_sigma
        if sigma == 0:
            return None
        if self.round_offsets is not None:
            return self.round_offsets[which][rows][:, None, :]
        return np.stack(
            [self.noise_rngs[i].normal(0.0, sigma, shape_tail) for i in rows]
        )

    def _estimate(self, rows: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Estimate P(2, 2) per execution from ``(len(rows), E, 4, 4)`` probabilities."""
        port = DETECTION_PORT
        if self.settings.counts is None:
            return probabilities[..., port, port]
        counts = np.stack(
            [
                self.count_rngs[i].poisson(self.settings.counts * np.clip(p, 0.0, None))
                for i, p in zip(rows, probabilities, strict=True)
            ]
        )
        totals = counts.sum(axis=(-2, -1))
        hits = counts[..., port, port]
        return np.divide(hits, totals, out=np.zeros(hits.shape), where=totals > 0)

    def measure_generated(
        self, g: np.ndarray, d: np.ndarray, index: np.ndarray | None = None
    ) -> np.ndarray:
        """Estimate tr[M rho] for phases ``(R, E, 30)`` and ``(R, E, 12)``.

        ``index`` lists the R seed rows measured (default: all seeds).
        """
        rows = self._rows(index)
        r, e = g.shape[:2]
        g_noise = self._noise(rows, (e, G_SIZE), 0)
        d_noise = self._noise(rows, (e, D_SIZE), 1)
        if g_noise is not None:
            g = g + g_noise
            d = d + d_noise
        g_flat, d_flat = g.reshape(r * e, G_SIZE), d.reshape(r * e, D_SIZE)
        amplitudes = generated_amplitudes(
            g_flat[:, :GENERATOR_NUM_PHASES], g_flat[:, GENERATOR_NUM_PHASES:]
        )
        probabilities = output_distribution(
            amplitudes,
            d_flat[:, :DISCRIMINATOR_NUM_PHASES],
            d_flat[:, DISCRIMINATOR_NUM_PHASES:],
        )
        return self._estimate(rows, probabilities.reshape(r, e, 4, 4))

    def measure_target(
        self, d: np.ndarray, index: np.ndarray | None = None
    ) -> np.ndarray:
        """Estimate tr[M tau] for discriminator phases ``(R, E, 12)``."""
        rows = self._rows(index)
        r, e = d.shape[:2]
        exact = self.settings.tau_measurement == "exact"
        if not exact:
            d_noise = self._noise(rows, (e, D_SIZE), 1)
            if d_noise is not None:
                d = d + d_noise
        d_flat = d.reshape(r * e, D_SIZE)
        amplitudes = np.repeat(self.targets[rows], e, axis=0)
        probabilities = output_distribution(
            amplitudes,
            d_flat[:, :DISCRIMINATOR_NUM_PHASES],
            d_flat[:, DISCRIMINATOR_NUM_PHASES:],
        ).reshape(r, e, 4, 4)
        if exact:
            return probabilities[..., DETECTION_PORT, DETECTION_PORT]
        return self._estimate(rows, probabilities)


def _shifted(params: np.ndarray, shift: float) -> np.ndarray:
    """Return ``(S, 1 + 2n, n)``: the parameters, then +shift and -shift on each one."""
    s, n = params.shape
    out = np.repeat(params[:, None, :], 1 + 2 * n, axis=1)
    index = np.arange(n)
    out[:, 1 + 2 * index, index] += shift
    out[:, 2 + 2 * index, index] -= shift
    return out


def _shift_gradient(values: np.ndarray, shift: float) -> np.ndarray:
    """Gradient from ``(S, 1 + 2n)`` shifted values (column 0 is unshifted)."""
    return (values[:, 1::2] - values[:, 2::2]) / (2.0 * np.sin(shift))


def _exact_d(
    g: np.ndarray, d: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return d and the fidelity for intended phases ``(S, 30)``, ``(S, 12)``."""
    amplitudes = generated_amplitudes(
        g[:, :GENERATOR_NUM_PHASES], g[:, GENERATOR_NUM_PHASES:]
    )
    ds, di = d[:, :DISCRIMINATOR_NUM_PHASES], d[:, DISCRIMINATOR_NUM_PHASES:]
    port = DETECTION_PORT
    a = output_distribution(amplitudes, ds, di)[:, port, port]
    b = output_distribution(targets, ds, di)[:, port, port]
    return np.abs(a - b), root_fidelity(amplitudes, targets)


def _discriminator_turn(
    g: np.ndarray, d: np.ndarray, hardware: _Hardware, settings: QganSettings
) -> np.ndarray:
    """Gradient ascent of d over the discriminator phases ``(S, 12)``."""
    shift, lr = settings.shift, settings.learning_rate
    for _ in range(settings.steps_per_turn):
        d_exec = _shifted(d, shift)
        g_exec = np.repeat(g[:, None, :], d_exec.shape[1], axis=1)
        a = hardware.measure_generated(g_exec, d_exec)
        b = hardware.measure_target(d_exec)
        sign = np.sign(a[:, 0] - b[:, 0])[:, None]
        d = d + lr * sign * (_shift_gradient(a, shift) - _shift_gradient(b, shift))
    return d


def _generator_turn(
    g: np.ndarray,
    d: np.ndarray,
    free: np.ndarray,
    hardware: _Hardware,
    settings: QganSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent of d over the free generator phases ``(S, 30)``.

    Returns the new phases and, per seed, the last d measured in the turn.
    """
    shift, lr, stop = settings.shift, settings.learning_rate, settings.generator_stop_d
    g = g.copy()
    active = np.ones(g.shape[0], dtype=bool)
    last_measured = np.full(g.shape[0], np.nan)
    for _ in range(settings.steps_per_turn):
        rows = np.nonzero(active)[0]
        if rows.size == 0:
            break
        g_exec = _shifted(g[rows], shift)
        d_exec = np.repeat(d[rows][:, None, :], g_exec.shape[1], axis=1)
        a = hardware.measure_generated(g_exec, d_exec, rows)
        b = hardware.measure_target(d_exec[:, :1], rows)
        measured = np.abs(a[:, 0] - b[:, 0])
        last_measured[rows] = measured
        moving = np.ones(rows.size, dtype=bool)
        if stop is not None:
            moving = measured > stop
            active[rows[~moving]] = False
        sign = np.sign(a[:, 0] - b[:, 0])[:, None]
        step = lr * (moving[:, None] * free[rows]) * sign * _shift_gradient(a, shift)
        g[rows] = g[rows] - step
    return g, last_measured


def noise_averaged_fidelity(
    g: np.ndarray,
    targets: np.ndarray,
    sigma: float,
    draws: int,
    rngs: list[np.random.Generator],
) -> np.ndarray:
    """Average root fidelity when every generator phase gets N(0, sigma^2) noise.

    Parameters
    ----------
    g : numpy.ndarray
        Intended generator phases, shape ``(S, 30)``.
    targets : numpy.ndarray
        Target amplitude matrices, shape ``(S, 4, 4)``.
    sigma : float
        Phase-noise standard deviation in radians.
    draws : int
        Noise draws per seed.
    rngs : list of numpy.random.Generator
        One stream per seed.

    Returns
    -------
    numpy.ndarray
        Shape ``(S,)``.
    """
    s = g.shape[0]
    if sigma == 0:
        amplitudes = generated_amplitudes(
            g[:, :GENERATOR_NUM_PHASES], g[:, GENERATOR_NUM_PHASES:]
        )
        return root_fidelity(amplitudes, targets)
    noisy = g[:, None, :] + np.stack(
        [rng.normal(0.0, sigma, (draws, G_SIZE)) for rng in rngs]
    )
    flat = noisy.reshape(s * draws, G_SIZE)
    amplitudes = generated_amplitudes(
        flat[:, :GENERATOR_NUM_PHASES], flat[:, GENERATOR_NUM_PHASES:]
    ).reshape(s, draws, 4, 4)
    return root_fidelity(amplitudes, targets[:, None]).mean(axis=1)


def train_qgan(inputs: SeedInputs, settings: QganSettings) -> QganResult:
    """Play the QGAN game for every seed of ``inputs``.

    Parameters
    ----------
    inputs : SeedInputs
        Targets, discriminator start and stuck sets.
    settings : QganSettings
        Training and hardware settings.

    Returns
    -------
    QganResult
    """
    s = len(inputs.seeds)
    hardware = _Hardware(inputs, settings)
    stuck = inputs.stuck_mask.reshape(s, G_SIZE)
    free = (~stuck).astype(float)
    g = np.tile(np.concatenate([generator_identity_phases()] * 2), (s, 1))
    g = np.where(stuck, inputs.stuck_values.reshape(s, G_SIZE), g)
    d = inputs.discriminator_init.reshape(s, D_SIZE).copy()
    fixed_rngs = [seed_stream(x, "fixed_discriminator") for x in inputs.seeds.tolist()]
    eval_rngs = [seed_stream(x, "evaluation_noise") for x in inputs.seeds.tolist()]
    targets = inputs.target_amplitudes

    rounds = settings.rounds
    fidelity = np.empty((rounds + 1, s))
    d_measured = np.empty((rounds, s))
    d_exact = np.empty((rounds, s))
    _, fidelity[0] = _exact_d(g, d, targets)
    result = QganResult(fidelity=fidelity, d_measured=d_measured, d_exact=d_exact)

    logger.info(
        "TRAINING_STARTED | rounds=%d | steps_per_turn=%d | generator_stop_d=%s "
        "| learning_rate=%.6g rad | shift=%.6g rad | phase_sigma=%.6g rad "
        "| counts=%s | discriminator=%s | seeds=%d | initial_mean_fidelity=%.6f",
        rounds,
        settings.steps_per_turn,
        settings.generator_stop_d,
        settings.learning_rate,
        settings.shift,
        settings.phase_sigma,
        settings.counts,
        settings.discriminator,
        s,
        fidelity[0].mean(),
    )
    for round_index in range(1, rounds + 1):
        hardware.start_round()
        if settings.discriminator == "random_fixed":
            d = np.stack([rng.uniform(0.0, 2.0 * np.pi, D_SIZE) for rng in fixed_rngs])
        else:
            d = _discriminator_turn(g, d, hardware, settings)
        g, d_measured[round_index - 1] = _generator_turn(g, d, free, hardware, settings)
        d_exact[round_index - 1], fidelity[round_index] = _exact_d(g, d, targets)
        logger.info(
            "TRAIN_ROUND_COMPLETED | round=%d/%d | mean_fidelity=%.6f "
            "| min_fidelity=%.6f | mean_d_measured=%.6f | mean_d_exact=%.6f",
            round_index,
            rounds,
            fidelity[round_index].mean(),
            fidelity[round_index].min(),
            d_measured[round_index - 1].mean(),
            d_exact[round_index - 1].mean(),
        )
        if round_index % settings.eval_every == 0 or round_index == rounds:
            logger.info(
                "EVALUATION_STARTED | round=%d | noise_draws=%d",
                round_index,
                settings.noise_draws,
            )
            averaged = noise_averaged_fidelity(
                g,
                targets,
                settings.phase_sigma,
                settings.noise_draws,
                eval_rngs,
            )
            result.noise_averaged[round_index] = averaged
            logger.info(
                "EVALUATION_COMPLETED | round=%d | mean_fidelity=%.6f "
                "| mean_noise_averaged_fidelity=%.6f",
                round_index,
                fidelity[round_index].mean(),
                averaged.mean(),
            )
    result.generator_phases = g.reshape(s, 2, GENERATOR_NUM_PHASES)
    result.discriminator_phases = d.reshape(s, 2, DISCRIMINATOR_NUM_PHASES)
    return result
