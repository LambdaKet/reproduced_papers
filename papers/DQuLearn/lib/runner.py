"""DQuLearn reproduction runner: one experiment per run.

``experiment.kind`` selects what a run does:

- ``runtime``: one client trains QuClassi for ``training.epochs`` epochs through
  the configured executor and records the wall-clock of each epoch (claim
  worker-scaling). ``execution.backend`` is ``manager`` (DQuLearn's co-Manager
  and local worker processes) or one of the baselines ``pool``, ``batched`` and
  ``local``;
- ``multi_tenant``: several clients train one epoch each through one co-Manager,
  concurrently or one after another, and each job's completion time is recorded
  (claim multi-tenant-gain);
- ``accuracy``: QuClassi is trained on a digit pair and evaluated on the
  validation and test splits (claim accuracy-parity);
- ``classical``: logistic regression on the same features (context baseline).

Artifacts written to the run directory (paths relative to it): ``metrics.json``,
``run_status.json`` and, with the manager backend, ``worker-<i>.log`` and
``worker-<i>.stderr.log``. The shared runtime writes ``run.log`` and
``config_snapshot.json``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .circuits import QuClassiLayout
from .classical import logistic_regression_baseline
from .cluster import LocalCluster, ManagerExecutor, WorkerSpec
from .comanager import CoManager
from .data import SPLITS, PairDataset, SplitArrays, prepare_pair_dataset
from .execution import (
    BatchedExecutor,
    CircuitTiming,
    ExecutionSettings,
    LocalExecutor,
    ProcessPoolCircuitExecutor,
    quiet_qiskit_logging,
)
from .experiment_logging import git_state, sha256, utc_now, write_json
from .quclassi import QuClassiModel, resolve_circuit_budget

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_KINDS = ("runtime", "multi_tenant", "accuracy", "classical")
BACKENDS = ("manager", "local", "pool", "batched")
MULTI_TENANT_MODES = ("multi", "single", "sequential")
N_CLASSES = 2

SELECTION = {
    "runtime": {
        "name": "epoch wall-clock (no model selection)",
        "direction": "minimize",
        "split": "none",
        "checkpoint": "none",
    },
    "multi_tenant": {
        "name": "job completion time (no model selection)",
        "direction": "minimize",
        "split": "none",
        "checkpoint": "none",
    },
    "accuracy": {
        "name": "learning rate across runs, by validation accuracy",
        "direction": "maximize",
        "split": "validation",
        "checkpoint": "final epoch",
    },
    "classical": {
        "name": "C by validation accuracy",
        "direction": "maximize",
        "split": "validation",
        "checkpoint": "fitted model",
    },
}


def _execution_settings(execution: dict) -> ExecutionSettings:
    return ExecutionSettings(
        shots=int(execution["shots"]),
        exact=bool(execution["exact"]),
        compile_mode=execution["compile_mode"],
    )


def _timing_summary(timings: list[CircuitTiming]) -> dict:
    """Median and mean of each timing component, in seconds, plus per-worker counts."""
    if not timings:
        return {"circuits": 0}
    values = np.array(
        [[t.queued, t.round_trip, t.worker_wait, t.execution] for t in timings]
    )
    summary: dict = {"circuits": len(timings)}
    for column, name in enumerate(("queued", "round_trip", "worker_wait", "execution")):
        summary[f"{name}_median"] = float(np.median(values[:, column]))
        summary[f"{name}_mean"] = float(np.mean(values[:, column]))
    summary["execution_total"] = float(values[:, 3].sum())
    summary["circuits_per_worker"] = dict(Counter(t.worker for t in timings))
    return summary


def _worker_spec(max_qubits, execution: dict) -> WorkerSpec:
    return WorkerSpec(
        int(max_qubits),
        int(execution["cpu_slots"]),
        float(execution["latency_seconds"]),
    )


@contextmanager
def _manager(
    cfg: dict, run_dir: Path, specs: list[WorkerSpec], min_qubits: int, tenancy: str
):
    """Start the workers and the co-Manager over them."""
    execution = cfg["execution"]
    connections = [
        1 if tenancy == "single" else max(1, spec.max_qubits // min_qubits)
        for spec in specs
    ]
    with LocalCluster(
        specs,
        _execution_settings(execution),
        run_dir,
        connections_per_worker=connections,
    ) as cluster:
        manager = CoManager(
            cluster.handles,
            policy=execution["policy"],
            tenancy=tenancy,
            heartbeat_seconds=float(execution["heartbeat_seconds"]),
            missed_heartbeat_limit=int(execution["missed_heartbeat_limit"]),
            max_parallel_calls=sum(connections),
        )
        try:
            yield cluster, manager
        finally:
            manager.close()


@contextmanager
def _single_client_executor(cfg: dict, run_dir: Path, layout: QuClassiLayout):
    """Executor of one client, and the resources behind it (cluster, manager)."""
    execution = cfg["execution"]
    backend = execution["backend"]
    if backend not in BACKENDS:
        raise ValueError(
            f"execution.backend must be one of {BACKENDS}, got {backend!r}"
        )
    settings = _execution_settings(execution)
    if backend == "manager":
        worker_qubits = execution["worker_qubits"] or [layout.n_qubits] * int(
            execution["workers"]
        )
        specs = [_worker_spec(q, execution) for q in worker_qubits]
        with _manager(cfg, run_dir, specs, layout.n_qubits, execution["tenancy"]) as (
            cluster,
            manager,
        ):
            yield (
                ManagerExecutor(manager, "client-1"),
                {"cluster": cluster, "manager": manager},
            )
        return
    if backend == "pool":
        executor = ProcessPoolCircuitExecutor(settings, int(execution["workers"]))
    elif backend == "batched":
        executor = BatchedExecutor(settings)
    else:
        executor = LocalExecutor(settings)
    try:
        yield executor, {}
    finally:
        executor.close()


def _warm_up(executor, model: QuClassiModel, split: SplitArrays, run_seed: int) -> int:
    """Run one bank before timing, then forget its timings and assignments.

    Returns the number of warm-up circuits.
    """
    tasks = model.warm_up_tasks(split.angles[0], run_seed)
    executor.run(tasks)
    executor.timings.clear()
    if isinstance(executor, ManagerExecutor):
        executor.manager.reset_assignment_counts()
    return len(tasks)


def _prepare_data(
    cfg: dict, feature_counts: set[int], status: dict, run_dir: Path
) -> dict[int, PairDataset]:
    """Load the pair once per feature count, record split sizes, log DATASET_READY."""
    datasets = {
        n_features: prepare_pair_dataset(
            cfg["data"], n_features, int(cfg["seed"]), cfg.get("data_root")
        )
        for n_features in sorted(feature_counts)
    }
    first = datasets[min(datasets)]
    status["dataset"]["splits"] = dict(first.meta["counts"])
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | pair=%s | features=%s | counts=%s",
        first.meta["pair"],
        sorted(datasets),
        first.meta["counts"],
    )
    return datasets


def _check_budget(
    split: SplitArrays, layout: QuClassiLayout, budget: int | None
) -> None:
    if budget is not None and 2 * layout.n_parameters * len(split) < budget:
        raise ValueError(
            f"{len(split)} training samples give at most {2 * layout.n_parameters * len(split)} "
            f"circuits, fewer than the budget of {budget}; raise data.subsample.size"
        )


def _manager_state(resources: dict) -> dict:
    manager = resources.get("manager")
    if manager is None:
        return {}
    return {
        "workers": manager.snapshot(),
        "heartbeat_rounds": manager.heartbeat_rounds,
        "startup_seconds": resources["cluster"].startup_seconds,
    }


def _run_runtime(cfg: dict, run_dir: Path, status: dict) -> dict:
    layout = QuClassiLayout(int(cfg["model"]["qubits"]), int(cfg["model"]["layers"]))
    training, execution = cfg["training"], cfg["execution"]
    dataset = _prepare_data(cfg, {layout.n_features}, status, run_dir)[
        layout.n_features
    ]
    budget = resolve_circuit_budget(training["circuits_per_epoch"], layout)
    _check_budget(dataset.train, layout, budget)
    model = QuClassiModel(
        layout, N_CLASSES, cfg["model"]["init_range"], int(cfg["seed"])
    )
    epochs = int(training["epochs"])
    with _single_client_executor(cfg, run_dir, layout) as (executor, resources):
        warm_up_circuits = _warm_up(executor, model, dataset.train, int(cfg["seed"]))
        logger.info(
            "TRAINING_STARTED | kind=runtime | backend=%s | workers=%s | qubits=%d | layers=%d "
            "| budget=%s | epochs=%d",
            execution["backend"],
            execution["workers"],
            layout.n_qubits,
            layout.n_layers,
            budget,
            epochs,
        )
        history = []
        for epoch_index in range(epochs):
            stats = model.train_epoch(
                dataset.train,
                executor,
                epoch_index=epoch_index,
                lr=float(training["lr"]),
                update_rule=training["update_rule"],
                shift_schedule=training["shift_schedule"],
                dispatch_unit=training["dispatch_unit"],
                circuit_budget=budget,
                run_seed=int(cfg["seed"]),
            )
            history.append(
                {
                    "epoch": epoch_index + 1,
                    "circuits": stats.circuits,
                    "samples": stats.samples,
                    "seconds": stats.seconds,
                    "circuits_per_second": stats.circuits / stats.seconds,
                }
            )
            logger.info(
                "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | circuits=%d | samples=%d | seconds=%.3f "
                "| circuits_per_second=%.2f",
                epoch_index + 1,
                epochs,
                stats.circuits,
                stats.samples,
                stats.seconds,
                stats.circuits / stats.seconds,
            )
        logger.info("EVALUATION_STARTED | measurement=epoch wall-clock")
        timing = _timing_summary(executor.timings)
        if execution["backend"] == "manager":
            slots = len(resources["cluster"].specs) * int(execution["cpu_slots"])
        elif execution["backend"] == "pool":
            slots = int(execution["workers"])
        else:
            slots = 1
        epoch_seconds = history[0]["seconds"]
        utilization = timing.get("execution_total", 0.0) / (
            sum(record["seconds"] for record in history) * slots
        )
        metrics = {
            "experiment": "runtime",
            "backend": execution["backend"],
            "workers": int(execution["workers"]),
            "cpu_slots": int(execution["cpu_slots"]),
            "latency_seconds": float(execution["latency_seconds"]),
            "compile_mode": execution["compile_mode"],
            "shots": int(execution["shots"]),
            "exact": bool(execution["exact"]),
            "qubits": layout.n_qubits,
            "layers": layout.n_layers,
            "parameters_per_class": layout.n_parameters,
            "circuit_budget": budget,
            "dispatch_unit": training["dispatch_unit"],
            "warm_up_circuits": warm_up_circuits,
            "epochs": history,
            "epoch_seconds": epoch_seconds,
            "circuits_per_second": history[0]["circuits_per_second"],
            "worker_utilization": utilization,
            "timing_seconds": timing,
            "manager": _manager_state(resources),
            "dataset": dataset.meta,
        }
        logger.info(
            "EVALUATION_COMPLETED | epoch_seconds=%.3f | circuits_per_second=%.2f "
            "| worker_utilization=%.3f",
            epoch_seconds,
            metrics["circuits_per_second"],
            utilization,
        )
    return metrics


def _run_multi_tenant(cfg: dict, run_dir: Path, status: dict) -> dict:
    tenant_cfg, training, execution = (
        cfg["multi_tenant"],
        cfg["training"],
        cfg["execution"],
    )
    mode = tenant_cfg["mode"]
    if mode not in MULTI_TENANT_MODES:
        raise ValueError(f"multi_tenant.mode must be one of {MULTI_TENANT_MODES}")
    if execution["backend"] != "manager" or not execution["worker_qubits"]:
        raise ValueError(
            "multi_tenant needs the manager backend and execution.worker_qubits"
        )
    jobs = list(tenant_cfg["jobs"])
    names = [job["name"] for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("multi_tenant job names must be unique")
    layouts = {
        job["name"]: QuClassiLayout(int(job["qubits"]), int(job["layers"]))
        for job in jobs
    }
    datasets = _prepare_data(
        cfg, {layout.n_features for layout in layouts.values()}, status, run_dir
    )
    budgets = {}
    for job in jobs:
        layout = layouts[job["name"]]
        budgets[job["name"]] = resolve_circuit_budget(job["circuits"], layout)
        _check_budget(datasets[layout.n_features].train, layout, budgets[job["name"]])
    seed = int(cfg["seed"])
    order = list(names)
    if tenant_cfg["shuffle_order"]:
        order = [order[i] for i in np.random.default_rng(seed).permutation(len(order))]
    job_seeds = {name: seed + 1000 * (index + 1) for index, name in enumerate(names)}
    specs = [_worker_spec(q, execution) for q in execution["worker_qubits"]]
    tenancy = "single" if mode == "single" else "multi"
    min_qubits = min(layout.n_qubits for layout in layouts.values())

    with _manager(cfg, run_dir, specs, min_qubits, tenancy) as (cluster, manager):
        models = {
            name: QuClassiModel(
                layouts[name], N_CLASSES, cfg["model"]["init_range"], job_seeds[name]
            )
            for name in names
        }
        executors = {name: ManagerExecutor(manager, name) for name in names}
        warm_up_circuits = sum(
            _warm_up(
                executors[name],
                models[name],
                datasets[layouts[name].n_features].train,
                job_seeds[name],
            )
            for name in order
        )
        logger.info(
            "TRAINING_STARTED | kind=multi_tenant | mode=%s | policy=%s | order=%s "
            "| worker_qubits=%s | budgets=%s",
            mode,
            execution["policy"],
            order,
            execution["worker_qubits"],
            budgets,
        )
        results: dict[str, dict] = {}
        errors: list[BaseException] = []
        common_start = time.perf_counter()

        def run_job(name: str) -> None:
            try:
                job_start = time.perf_counter()
                stats = models[name].train_epoch(
                    datasets[layouts[name].n_features].train,
                    executors[name],
                    epoch_index=0,
                    lr=float(training["lr"]),
                    update_rule=training["update_rule"],
                    shift_schedule=training["shift_schedule"],
                    dispatch_unit=training["dispatch_unit"],
                    circuit_budget=budgets[name],
                    run_seed=job_seeds[name],
                )
                job_end = time.perf_counter()
                results[name] = {
                    "qubits": layouts[name].n_qubits,
                    "layers": layouts[name].n_layers,
                    "circuits": stats.circuits,
                    "samples": stats.samples,
                    "start_seconds": job_start - common_start,
                    "completion_seconds": job_end - common_start,
                    "duration_seconds": job_end - job_start,
                    "circuits_per_second": stats.circuits / (job_end - job_start),
                    "timing_seconds": _timing_summary(executors[name].timings),
                }
                logger.info(
                    "TRAIN_JOB_COMPLETED | job=%s | circuits=%d | duration=%.2f | completion=%.2f",
                    name,
                    stats.circuits,
                    job_end - job_start,
                    job_end - common_start,
                )
            except BaseException as error:  # noqa: BLE001 - re-raised by the caller
                errors.append(error)

        if mode == "sequential":
            for name in order:
                run_job(name)
                if errors:
                    break
        else:
            threads = [
                threading.Thread(target=run_job, args=(name,), name=f"client-{name}")
                for name in order
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if errors:
            raise errors[0]
        makespan = time.perf_counter() - common_start
        logger.info("EVALUATION_STARTED | measurement=job completion times")
        total_circuits = sum(result["circuits"] for result in results.values())
        metrics = {
            "experiment": "multi_tenant",
            "mode": mode,
            "tenancy": tenancy,
            "policy": execution["policy"],
            "worker_qubits": list(execution["worker_qubits"]),
            "cpu_slots": int(execution["cpu_slots"]),
            "latency_seconds": float(execution["latency_seconds"]),
            "compile_mode": execution["compile_mode"],
            "shots": int(execution["shots"]),
            "submission_order": order,
            "warm_up_circuits": warm_up_circuits,
            "jobs": {name: results[name] for name in names},
            "makespan_seconds": makespan,
            "system_circuits_per_second": total_circuits / makespan,
            "manager": _manager_state({"cluster": cluster, "manager": manager}),
        }
        logger.info(
            "EVALUATION_COMPLETED | makespan=%.2f | system_circuits_per_second=%.2f "
            "| completion=%s",
            makespan,
            metrics["system_circuits_per_second"],
            {name: round(results[name]["completion_seconds"], 2) for name in names},
        )
    return metrics


def _evaluation_key(epoch_index: int, split_name: str) -> int:
    return (epoch_index + 1) * 10 + SPLITS.index(split_name)


def _run_accuracy(cfg: dict, run_dir: Path, status: dict) -> dict:
    layout = QuClassiLayout(int(cfg["model"]["qubits"]), int(cfg["model"]["layers"]))
    training, execution, evaluation = (
        cfg["training"],
        cfg["execution"],
        cfg["evaluation"],
    )
    dataset = _prepare_data(cfg, {layout.n_features}, status, run_dir)[
        layout.n_features
    ]
    budget = resolve_circuit_budget(training["circuits_per_epoch"], layout)
    _check_budget(dataset.train, layout, budget)
    seed = int(cfg["seed"])
    model = QuClassiModel(layout, N_CLASSES, cfg["model"]["init_range"], seed)
    epochs = int(training["epochs"])

    def evaluate(split_name: str, epoch_index: int, executor) -> dict | None:
        split = dataset.split(split_name)
        if len(split) == 0:
            return None
        label = "final" if epoch_index == epochs else f"epoch={epoch_index + 1}"
        logger.info("EVALUATION_STARTED | split=%s | %s", split_name, label)
        result = model.evaluate(
            split,
            executor,
            run_seed=seed,
            evaluation_key=_evaluation_key(epoch_index, split_name),
        )
        logger.info(
            "EVALUATION_COMPLETED | split=%s | %s | accuracy=%.4f | loss=%.4f",
            split_name,
            label,
            result["accuracy"],
            result["loss"],
        )
        return result

    with _single_client_executor(cfg, run_dir, layout) as (executor, resources):
        logger.info(
            "TRAINING_STARTED | kind=accuracy | backend=%s | workers=%s | exact=%s | update=%s "
            "| shift=%s | lr=%g | epochs=%d",
            execution["backend"],
            execution["workers"],
            execution["exact"],
            training["update_rule"],
            training["shift_schedule"],
            float(training["lr"]),
            epochs,
        )
        history = []
        start = time.perf_counter()
        for epoch_index in range(epochs):
            stats = model.train_epoch(
                dataset.train,
                executor,
                epoch_index=epoch_index,
                lr=float(training["lr"]),
                update_rule=training["update_rule"],
                shift_schedule=training["shift_schedule"],
                dispatch_unit=training["dispatch_unit"],
                circuit_budget=budget,
                run_seed=seed,
            )
            record = {
                "epoch": epoch_index + 1,
                "circuits": stats.circuits,
                "seconds": stats.seconds,
                "mean_shifted_loss": stats.mean_shifted_loss,
            }
            logger.info(
                "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | circuits=%d | seconds=%.2f "
                "| mean_shifted_loss=%.4f",
                epoch_index + 1,
                epochs,
                stats.circuits,
                stats.seconds,
                stats.mean_shifted_loss,
            )
            for split_name in evaluation["each_epoch"]:
                result = evaluate(split_name, epoch_index, executor)
                if result is not None:
                    record[split_name] = result
            history.append(record)
        final = {}
        for split_name in evaluation["final"]:
            result = evaluate(split_name, epochs, executor)
            if result is not None:
                final[split_name] = result
        metrics = {
            "experiment": "accuracy",
            "backend": execution["backend"],
            "workers": int(execution["workers"]),
            "exact": bool(execution["exact"]),
            "shots": int(execution["shots"]),
            "compile_mode": execution["compile_mode"],
            "qubits": layout.n_qubits,
            "layers": layout.n_layers,
            "update_rule": training["update_rule"],
            "shift_schedule": training["shift_schedule"],
            "init_range": cfg["model"]["init_range"],
            "lr": float(training["lr"]),
            "epochs": epochs,
            "history": history,
            "final": final,
            "parameters": model.parameters.tolist(),
            "training_seconds": time.perf_counter() - start,
            "timing_seconds": _timing_summary(executor.timings),
            "manager": _manager_state(resources),
            "dataset": dataset.meta,
        }
    return metrics


def _run_classical(cfg: dict, run_dir: Path, status: dict) -> dict:
    n_features = QuClassiLayout(int(cfg["model"]["qubits"]), 1).n_features
    dataset = _prepare_data(cfg, {n_features}, status, run_dir)[n_features]
    c_grid = list(cfg["classical"]["c_grid"])
    logger.info(
        "TRAINING_STARTED | kind=classical | model=logistic regression | c_grid=%s",
        c_grid,
    )
    result = logistic_regression_baseline(dataset, c_grid, int(cfg["seed"]))
    for candidate in result["candidates"]:
        logger.info(
            "TRAIN_CANDIDATE_COMPLETED | C=%g | train_accuracy=%.4f | validation_accuracy=%.4f",
            candidate["C"],
            candidate["train_accuracy"],
            candidate["validation_accuracy"],
        )
    logger.info("EVALUATION_STARTED | split=test | selected_C=%g", result["selected_C"])
    logger.info(
        "EVALUATION_COMPLETED | split=test | selected_C=%g | accuracy=%.4f",
        result["selected_C"],
        result["test_accuracy"],
    )
    return {"experiment": "classical", "dataset": dataset.meta, **result}


RUNNERS = {
    "runtime": _run_runtime,
    "multi_tenant": _run_multi_tenant,
    "accuracy": _run_accuracy,
    "classical": _run_classical,
}


def _candidate(cfg: dict, kind: str) -> dict:
    """Settings that identify a run of this kind, for run_status.json."""
    execution, training = cfg["execution"], cfg["training"]
    if kind == "runtime":
        return {
            "backend": execution["backend"],
            "workers": execution["workers"],
            "qubits": cfg["model"]["qubits"],
            "layers": cfg["model"]["layers"],
            "compile_mode": execution["compile_mode"],
            "dispatch_unit": training["dispatch_unit"],
        }
    if kind == "multi_tenant":
        return {
            "mode": cfg["multi_tenant"]["mode"],
            "policy": execution["policy"],
            "worker_qubits": execution["worker_qubits"],
            "cpu_slots": execution["cpu_slots"],
            "compile_mode": execution["compile_mode"],
        }
    if kind == "accuracy":
        return {
            "backend": execution["backend"],
            "workers": execution["workers"],
            "exact": execution["exact"],
            "update_rule": training["update_rule"],
            "shift_schedule": training["shift_schedule"],
            "lr": training["lr"],
            "pair": cfg["data"]["pair"],
        }
    return {"model": "logistic regression", "pair": cfg["data"]["pair"]}


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the configured experiment and write its evidence files.

    Parameters
    ----------
    cfg : dict
        Resolved configuration from the shared runtime.
    run_dir : pathlib.Path
        Run directory created by the runtime (holds ``run.log`` and
        ``config_snapshot.json``).

    Raises
    ------
    ValueError
        If the experiment kind, backend or device is not supported.
    """
    run_dir = Path(run_dir)
    quiet_qiskit_logging()
    kind = cfg["experiment"]["kind"]
    if kind not in EXPERIMENT_KINDS:
        raise ValueError(
            f"experiment.kind must be one of {EXPERIMENT_KINDS}, got {kind!r}"
        )
    data = cfg["data"]
    config_path = run_dir / "config_snapshot.json"
    status = {
        "schema_version": 1,
        "run_id": cfg.get("run_id") or run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate") or _candidate(cfg, kind),
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "dataset": {
            "name": data["name"],
            "splits": {"train": None, "validation": None, "test": None},
            "preprocessing": "PCA fitted on all 60,000 MNIST training images; min-max "
            "scaling on the subsample; angles 2*asin(sqrt(x)) (QuClassi's code)",
            "subset": f"digits {list(data['pair'])}; {data['subsample']['mode']} "
            f"{data['subsample']['size']} training images; validation fraction "
            f"{data['validation_fraction']}",
        },
        "selection": SELECTION[kind],
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    try:
        start = time.perf_counter()
        logger.info(
            "RUN_STARTED | run_id=%s | kind=%s | seed=%s",
            status["run_id"],
            kind,
            cfg["seed"],
        )
        if cfg["device"] != "cpu":
            raise ValueError(
                f"this implementation simulates on CPU only; device must be 'cpu', got "
                f"{cfg['device']!r}"
            )
        metrics = RUNNERS[kind](cfg, run_dir, status)
        metrics["seconds_total"] = time.perf_counter() - start
        write_json(run_dir / "metrics.json", metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info(
            "RUN_COMPLETED | run_id=%s | seconds=%.1f",
            status["run_id"],
            metrics["seconds_total"],
        )
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
