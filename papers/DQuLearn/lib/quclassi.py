"""QuClassi training and inference, as DQuLearn runs it (Algorithm 1).

One learned state per class. As in QuClassi's code, a class state trains only
on its own class's samples: the loss of a sample is L = −log F, where F is the
swap-test fidelity between the class state and the sample (QuClassi Eq. 14 with
y = 1). For each parameter, one circuit shifted by +s and one by −s give
QuClassi's estimate g = ½[L(θ+s) − L(θ−s)] (LOG.md, Gradient method).

Update rules (LOG.md, Reproducibility Assessment):

- ``simultaneous``: all shifted circuits of a sample form one bank, DQuLearn's
  circuit bank (Algorithm 1, lines 12–22), and all parameters update together;
- ``sequential``: QuClassi's code updates each parameter right after its two
  circuits, so the next parameter's circuits see the updated value.

A sample is predicted as the class whose state has the highest normalized
fidelity to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .circuits import QuClassiLayout, build_fidelity_circuit, circuit_to_bytes
from .data import SplitArrays
from .execution import CircuitTask

UPDATE_RULES = ("simultaneous", "sequential")
SHIFT_SCHEDULES = ("fixed", "shrinking")
DISPATCH_UNITS = ("sample", "epoch")
INIT_UPPER_BOUNDS = {"pi": np.pi, "two_pi": 2.0 * np.pi}

# DQuLearn Sec. IV-C1: 1,440 / 2,880 / 4,320 circuits per epoch at 5 qubits and
# 2,016 / 4,032 / 6,048 at 7 qubits, for 1 / 2 / 3 layers.
PAPER_CIRCUITS_PER_LAYER = {5: 1440, 7: 2016}

# Seed streams keep training, evaluation and warm-up circuits from sharing seeds.
_TRAINING_STREAM = 0
_EVALUATION_STREAM = 1
_WARM_UP_STREAM = 2


def paper_circuits_per_epoch(layout: QuClassiLayout) -> int:
    """Circuits per epoch reported by the paper for this width and depth.

    Raises
    ------
    ValueError
        If the paper reports no count for this width.
    """
    if layout.n_qubits not in PAPER_CIRCUITS_PER_LAYER:
        raise ValueError(
            f"the paper reports no circuit count at {layout.n_qubits} qubits"
        )
    return PAPER_CIRCUITS_PER_LAYER[layout.n_qubits] * layout.n_layers


def resolve_circuit_budget(value, layout: QuClassiLayout) -> int | None:
    """Circuit budget of one epoch.

    Parameters
    ----------
    value : str or int or None
        ``"paper"`` for the paper's count, ``"all"`` or None for no budget, or
        an even number of circuits.
    layout : QuClassiLayout
        Circuit width and depth.

    Returns
    -------
    int or None
        Circuits per epoch, or None for one pass over the training split.

    Raises
    ------
    ValueError
        If a numeric budget is not an even number of at least 2 (every
        parameter needs a pair of shifted circuits).
    """
    if value is None or value == "all":
        return None
    if value == "paper":
        return paper_circuits_per_epoch(layout)
    budget = int(value)
    if budget < 2 or budget % 2:
        raise ValueError(
            f"circuits_per_epoch must be an even number >= 2, got {value!r}"
        )
    return budget


def circuit_seed(*keys: int) -> int:
    """Simulator seed of one circuit, derived from the keys that identify it."""
    return int(np.random.SeedSequence([int(key) for key in keys]).generate_state(1)[0])


def shift_for_epoch(schedule: str, epoch_index: int) -> float:
    """Parameter shift of an epoch (epochs counted from 0).

    ``fixed`` is DQuLearn's π/2 (Algorithm 1, lines 15 and 18); ``shrinking`` is
    QuClassi's code, π / (2√(epoch + 1)).
    """
    if schedule == "fixed":
        return np.pi / 2
    if schedule == "shrinking":
        return np.pi / (2.0 * np.sqrt(epoch_index + 1))
    raise ValueError(
        f"shift_schedule must be one of {SHIFT_SCHEDULES}, got {schedule!r}"
    )


@dataclass(frozen=True)
class EpochStats:
    """What one training epoch did.

    Parameters
    ----------
    circuits : int
        Circuits executed.
    samples : int
        Samples visited (the last one may be partial under a budget).
    seconds : float
        Wall-clock of the epoch.
    mean_shifted_loss : float
        Mean of −log F over the shifted circuits of the epoch (not the loss at θ).
    """

    circuits: int
    samples: int
    seconds: float
    mean_shifted_loss: float


class QuClassiModel:
    """Learned class states of QuClassi.

    Parameters
    ----------
    layout : QuClassiLayout
        Circuit width and depth.
    n_classes : int
        Number of class states.
    init_range : str
        ``pi`` (DQuLearn, Algorithm 1) or ``two_pi`` (QuClassi's code): initial
        parameters uniform in ``[0, π]`` or ``[0, 2π)``.
    seed : int
        Seed of the initialization.
    """

    def __init__(
        self, layout: QuClassiLayout, n_classes: int, init_range: str, seed: int
    ) -> None:
        if init_range not in INIT_UPPER_BOUNDS:
            raise ValueError(
                f"init_range must be one of {tuple(INIT_UPPER_BOUNDS)}, got {init_range!r}"
            )
        rng = np.random.default_rng(seed)
        self.layout = layout
        self.parameters = rng.uniform(
            0.0, INIT_UPPER_BOUNDS[init_range], size=(n_classes, layout.n_parameters)
        )

    def _task(
        self, parameters: np.ndarray, angles: np.ndarray, seed: int
    ) -> CircuitTask:
        circuit = build_fidelity_circuit(self.layout, parameters, angles)
        return CircuitTask(circuit_to_bytes(circuit), self.layout.n_qubits, seed)

    def _shifted_tasks(
        self,
        class_index: int,
        angles: np.ndarray,
        shift: float,
        parameter_indices,
        keys: tuple[int, ...],
    ) -> list[CircuitTask]:
        """Circuits shifted by +s then −s for each listed parameter."""
        base = self.parameters[class_index]
        tasks = []
        for index in parameter_indices:
            for sign_code, sign in ((0, 1.0), (1, -1.0)):
                shifted = base.copy()
                shifted[index] += sign * shift
                tasks.append(
                    self._task(shifted, angles, circuit_seed(*keys, index, sign_code))
                )
        return tasks

    def train_epoch(
        self,
        split: SplitArrays,
        executor,
        *,
        epoch_index: int,
        lr: float,
        update_rule: str,
        shift_schedule: str,
        dispatch_unit: str,
        circuit_budget: int | None,
        run_seed: int,
    ) -> EpochStats:
        """Train for one epoch, or until the circuit budget is spent.

        Parameters
        ----------
        split : SplitArrays
            Training samples, visited in order.
        executor : object
            Anything with ``run(tasks) -> list of fidelities``.
        epoch_index : int
            Epoch number, from 0 (sets the shrinking shift and the seeds).
        lr : float
            Learning rate.
        update_rule : str
            ``simultaneous`` or ``sequential``.
        shift_schedule : str
            ``fixed`` or ``shrinking``.
        dispatch_unit : str
            ``sample``: one bank per sample (Algorithm 1). ``epoch``: every
            sample's circuits at the epoch's starting parameters in one bank,
            then the mean gradient of each class is applied (a runtime variant;
            it turns the optimizer into full-batch gradient descent).
        circuit_budget : int or None
            Circuits allowed this epoch; None for one pass over the split.
        run_seed : int
            Seed from which every circuit's simulator seed is derived.

        Returns
        -------
        EpochStats
            Circuits, samples, wall-clock and mean shifted loss.
        """
        if update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        if dispatch_unit not in DISPATCH_UNITS:
            raise ValueError(f"dispatch_unit must be one of {DISPATCH_UNITS}")
        if dispatch_unit == "epoch" and update_rule != "simultaneous":
            raise ValueError("epoch dispatch needs the simultaneous update rule")
        shift = shift_for_epoch(shift_schedule, epoch_index)
        n_parameters = self.layout.n_parameters
        budget = (
            circuit_budget
            if circuit_budget is not None
            else 2 * n_parameters * len(split)
        )
        start = time.perf_counter()
        used = 0
        samples = 0
        losses: list[float] = []

        if dispatch_unit == "epoch":
            tasks: list[CircuitTask] = []
            owners: list[tuple[int, int]] = []
            for sample_index, (angles, label) in enumerate(
                zip(split.angles, split.labels)
            ):
                count = min(n_parameters, (budget - used) // 2)
                if count == 0:
                    break
                keys = (run_seed, _TRAINING_STREAM, epoch_index, sample_index)
                tasks += self._shifted_tasks(
                    int(label), angles, shift, range(count), keys
                )
                owners.append((int(label), count))
                used += 2 * count
                samples += 1
            pair_losses = -np.log(np.asarray(executor.run(tasks)))
            losses.extend(pair_losses.tolist())
            gradient_sum = np.zeros_like(self.parameters)
            visits = np.zeros(len(self.parameters))
            position = 0
            for class_index, count in owners:
                block = pair_losses[position : position + 2 * count]
                position += 2 * count
                gradient_sum[class_index, :count] += 0.5 * (block[0::2] - block[1::2])
                visits[class_index] += 1
            for class_index in np.flatnonzero(visits):
                self.parameters[class_index] -= (
                    lr * gradient_sum[class_index] / visits[class_index]
                )
        else:
            for sample_index, (angles, label) in enumerate(
                zip(split.angles, split.labels)
            ):
                if used >= budget:
                    break
                class_index = int(label)
                keys = (run_seed, _TRAINING_STREAM, epoch_index, sample_index)
                if update_rule == "simultaneous":
                    count = min(n_parameters, (budget - used) // 2)
                    bank = self._shifted_tasks(
                        class_index, angles, shift, range(count), keys
                    )
                    pair_losses = -np.log(np.asarray(executor.run(bank)))
                    self.parameters[class_index, :count] -= (
                        lr * 0.5 * (pair_losses[0::2] - pair_losses[1::2])
                    )
                    used += 2 * count
                    losses.extend(pair_losses.tolist())
                else:
                    for index in range(n_parameters):
                        if used >= budget:
                            break
                        bank = self._shifted_tasks(
                            class_index, angles, shift, [index], keys
                        )
                        pair_losses = -np.log(np.asarray(executor.run(bank)))
                        self.parameters[class_index, index] -= (
                            lr * 0.5 * (pair_losses[0] - pair_losses[1])
                        )
                        used += 2
                        losses.extend(pair_losses.tolist())
                samples += 1

        return EpochStats(
            circuits=used,
            samples=samples,
            seconds=time.perf_counter() - start,
            mean_shifted_loss=float(np.mean(losses)) if losses else float("nan"),
        )

    def warm_up_tasks(self, angles: np.ndarray, run_seed: int) -> list[CircuitTask]:
        """One sample's shifted circuits on its own seed stream, to run before timing.

        Executing them starts pool processes and pays first-call compilation
        costs outside the measured epoch; parameters are not updated.
        """
        keys = (run_seed, _WARM_UP_STREAM, 0, 0)
        return self._shifted_tasks(
            0, angles, np.pi / 2, range(self.layout.n_parameters), keys
        )

    def fidelities(
        self, angles: np.ndarray, executor, *, run_seed: int, evaluation_key: int
    ) -> np.ndarray:
        """Fidelity of every sample to every class state, shape ``(n, n_classes)``."""
        tasks = [
            self._task(
                self.parameters[class_index],
                sample,
                circuit_seed(
                    run_seed,
                    _EVALUATION_STREAM,
                    evaluation_key,
                    sample_index,
                    class_index,
                ),
            )
            for sample_index, sample in enumerate(angles)
            for class_index in range(len(self.parameters))
        ]
        values = np.asarray(executor.run(tasks))
        return values.reshape(len(angles), len(self.parameters))

    def evaluate(
        self, split: SplitArrays, executor, *, run_seed: int, evaluation_key: int
    ) -> dict:
        """Accuracy and mean loss on a split.

        Returns
        -------
        dict
            ``accuracy`` (argmax of the normalized fidelities, as QuClassi's
            code), ``loss`` (mean −log F to the sample's own class state) and
            ``samples``.
        """
        fidelities = self.fidelities(
            split.angles, executor, run_seed=run_seed, evaluation_key=evaluation_key
        )
        probabilities = fidelities / fidelities.sum(axis=1, keepdims=True)
        predictions = probabilities.argmax(axis=1)
        own = fidelities[np.arange(len(split)), split.labels]
        return {
            "accuracy": float(np.mean(predictions == split.labels)),
            "loss": float(np.mean(-np.log(own))),
            "samples": int(len(split)),
        }
