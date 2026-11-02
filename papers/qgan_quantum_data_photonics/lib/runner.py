"""Runner of the photonic QGAN reproduction: one condition, a block of seeds.

``experiment.kind`` selects the run:

- ``qgan``: the paper's adversarial game (``lib.qgan``) for one noise level,
  count budget and defect setting, on every seed of the block (one target per
  seed);
- ``oracle_ceiling``: our control, direct maximization of the true fidelity
  over the same generator circuits (``lib.ceiling``);
- ``shot_noise_tvd``: the Suppl. Fig. 9a sampling check (``lib.metrics``).

Artifacts written to the run directory (paths relative to it):
``run_status.json``, ``metrics.json`` and ``trajectories.npz`` (QGAN) or
``ceiling.npz`` (ceiling). The shared runtime writes ``run.log`` and
``config_snapshot.json``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .ceiling import oracle_ceiling
from .data import (
    DATASET_NAME,
    SeedInputs,
    build_dataloader,
    inputs_identity,
    paper_data_dir,
    seed_stream,
)
from .experiment_logging import git_state, sha256, utc_now, write_json
from .metrics import (
    convergence_round,
    final_fidelity,
    first_round_reaching,
    shot_noise_tvd,
    summarize,
)
from .qgan import QganSettings, noise_averaged_fidelity, train_qgan

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
KINDS = ("qgan", "oracle_ceiling", "shot_noise_tvd")


def qgan_settings(cfg: dict) -> QganSettings:
    """Build the QGAN settings from the ``training``, ``noise`` and ``evaluation`` blocks.

    Angles are given in units of pi in the configs, as in the paper.
    """
    training, noise, evaluation = cfg["training"], cfg["noise"], cfg["evaluation"]
    counts = noise["counts"]
    stop = training["generator_stop_d"]
    return QganSettings(
        rounds=int(training["rounds"]),
        steps_per_turn=int(training["steps_per_turn"]),
        learning_rate=float(training["learning_rate_pi"]) * np.pi,
        shift=float(training["shift_pi"]) * np.pi,
        phase_sigma=float(noise["phase_sigma_pi"]) * np.pi,
        counts=None if counts in (None, 0) else int(counts),
        noise_redraw=noise["redraw"],
        tau_measurement=noise["tau_measurement"],
        discriminator=training["discriminator"],
        generator_stop_d=None if stop is None else float(stop),
        eval_every=int(evaluation["eval_every"]),
        noise_draws=int(evaluation["noise_draws"]),
    )


def _inputs(cfg: dict) -> SeedInputs:
    targets, init, mask, values, seeds = build_dataloader(cfg, split="train")[0]
    return SeedInputs(
        seeds=seeds,
        targets=targets,
        discriminator_init=init,
        stuck_mask=mask,
        stuck_values=values,
    )


def _run_qgan(cfg: dict, inputs: SeedInputs, run_dir: Path) -> dict:
    settings = qgan_settings(cfg)
    evaluation = cfg["evaluation"]
    window = int(evaluation["final_window"])
    backend = cfg["backend"]
    hardware, hardware_fields = None, {"simulator": "numpy (lib/circuits.py)"}
    if backend["name"] == "merlin":
        # Imported here so the numpy reproduction does not depend on MerLin at import time.
        from .qgan_merlin import MerlinHardware

        hardware = MerlinHardware(inputs, settings, backend["merlin_noise"])
        hardware_fields = hardware.hardware_fields(settings.counts)
        logger.info("BACKEND_READY | merlin | noise=%s", backend["merlin_noise"])
    elif backend["name"] != "numpy":
        raise ValueError(
            f"backend.name must be 'numpy' or 'merlin', got {backend['name']!r}"
        )
    result = train_qgan(inputs, settings, hardware)
    fidelity = result.fidelity
    final = final_fidelity(fidelity, window)
    convergence = convergence_round(
        fidelity, window, float(evaluation["convergence_tolerance"])
    )
    mean_curve = fidelity.mean(axis=1)
    last_round = max(result.noise_averaged)
    arrays_path = run_dir / "trajectories.npz"
    np.savez_compressed(
        arrays_path,
        seeds=inputs.seeds,
        fidelity=fidelity,
        d_measured=result.d_measured,
        d_exact=result.d_exact,
        noise_averaged_rounds=np.array(sorted(result.noise_averaged)),
        noise_averaged=np.stack(
            [result.noise_averaged[r] for r in sorted(result.noise_averaged)]
        ),
        generator_phases=result.generator_phases,
        discriminator_phases=result.discriminator_phases,
        stuck_mask=inputs.stuck_mask,
    )
    logger.info("ARTIFACT_WRITTEN | path=%s", arrays_path.name)
    threshold = float(evaluation["fidelity_threshold"])
    return {
        "settings": {
            "rounds": settings.rounds,
            "steps_per_turn": settings.steps_per_turn,
            "learning_rate_rad": settings.learning_rate,
            "shift_rad": settings.shift,
            "phase_sigma_rad": settings.phase_sigma,
            "counts": settings.counts,
            "noise_redraw": settings.noise_redraw,
            "tau_measurement": settings.tau_measurement,
            "discriminator": settings.discriminator,
            "generator_stop_d": settings.generator_stop_d,
        },
        "backend": backend,
        "hardware": hardware_fields,
        "fidelity_metric": "root fidelity |<g|tau>| with the intended generator phases",
        "initial_fidelity": summarize(fidelity[0]),
        "final_fidelity": summarize(final),
        "final_fidelity_per_seed": final.tolist(),
        "final_window_rounds": window,
        "noise_averaged_final_fidelity": summarize(result.noise_averaged[last_round]),
        "convergence_round": summarize(convergence),
        "convergence_round_per_seed": convergence.tolist(),
        "mean_fidelity_first_reaches": {
            "threshold": threshold,
            "round": first_round_reaching(mean_curve, threshold),
        },
        "mean_fidelity_by_round": mean_curve.tolist(),
        "std_fidelity_by_round": fidelity.std(axis=1, ddof=1).tolist()
        if fidelity.shape[1] > 1
        else [0.0] * fidelity.shape[0],
        "final_d_measured": summarize(result.d_measured[-window:].mean(axis=0)),
        "final_d_exact": summarize(result.d_exact[-window:].mean(axis=0)),
        "artifacts": {"trajectories": arrays_path.name},
    }


def _run_ceiling(cfg: dict, inputs: SeedInputs, run_dir: Path) -> dict:
    ceiling_cfg = cfg["ceiling"]
    best, phases = oracle_ceiling(
        inputs,
        restarts=int(ceiling_cfg["restarts"]),
        iterations=int(ceiling_cfg["iterations"]),
        learning_rate=float(ceiling_cfg["learning_rate"]),
        device=cfg["device"],
    )
    draws = int(ceiling_cfg["noise_draws"])
    rngs = [seed_stream(x, "ceiling_noise") for x in inputs.seeds.tolist()]
    flat = phases.reshape(len(inputs.seeds), -1)
    averaged = {}
    for sigma_pi in ceiling_cfg["phase_sigmas_pi"]:
        logger.info(
            "EVALUATION_STARTED | ceiling | phase_sigma=%.3gpi | draws=%d",
            sigma_pi,
            draws,
        )
        values = noise_averaged_fidelity(
            flat, inputs.target_amplitudes, float(sigma_pi) * np.pi, draws, rngs
        )
        averaged[str(sigma_pi)] = values
        logger.info(
            "EVALUATION_COMPLETED | ceiling | phase_sigma=%.3gpi "
            "| mean_noise_averaged_fidelity=%.6f",
            sigma_pi,
            values.mean(),
        )
    arrays_path = run_dir / "ceiling.npz"
    np.savez_compressed(
        arrays_path,
        seeds=inputs.seeds,
        best_fidelity=best,
        best_phases=phases,
        stuck_mask=inputs.stuck_mask,
        **{f"noise_averaged_sigma_{k}pi": v for k, v in averaged.items()},
    )
    logger.info("ARTIFACT_WRITTEN | path=%s", arrays_path.name)
    return {
        "fidelity_metric": "root fidelity |<g|tau>|, best over restarts",
        "best_fidelity": summarize(best),
        "best_fidelity_per_seed": best.tolist(),
        "noise_averaged_best_fidelity": {k: summarize(v) for k, v in averaged.items()},
        "noise_averaged_best_fidelity_per_seed": {
            k: v.tolist() for k, v in averaged.items()
        },
        "artifacts": {"ceiling": arrays_path.name},
    }


def _run_tvd(cfg: dict) -> dict:
    tvd_cfg = cfg["shot_noise_tvd"]
    logger.info(
        "EVALUATION_STARTED | shot_noise_tvd | levels=%s | repeats=%d",
        tvd_cfg["count_levels"],
        tvd_cfg["repeats"],
    )
    rng = seed_stream(int(cfg["seeds"]["start"]), "counts")
    tvd = shot_noise_tvd(
        [int(x) for x in tvd_cfg["count_levels"]], int(tvd_cfg["repeats"]), rng
    )
    logger.info("EVALUATION_COMPLETED | shot_noise_tvd | mean_tvd=%s", tvd)
    return {
        "metric": "mean total variation distance, uniform 16-outcome distribution",
        "repeats": int(tvd_cfg["repeats"]),
        "mean_tvd_by_count_level": {str(k): v for k, v in tvd.items()},
    }


def _dataset_block(cfg: dict) -> dict:
    kind = cfg["experiment"]["kind"]
    start, count = int(cfg["seeds"]["start"]), int(cfg["seeds"]["count"])
    if kind == "shot_noise_tvd":
        return {
            "name": "uniform_16_outcome_distribution",
            "splits": {"evaluation": "count levels in shot_noise_tvd.count_levels"},
            "preprocessing": "none (generated)",
            "subset": f"counts stream of seed {start}",
        }
    return {
        "name": DATASET_NAME,
        "splits": {"train": count},
        "preprocessing": "none (generated per seed; see lib/data.py)",
        "subset": f"seeds {start}-{start + count - 1}; defects {cfg['defects']}",
        "data_dir": str(paper_data_dir(cfg)),
    }


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run one configured condition and write the evidence files.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``configs/defaults.json`` plus overlays).
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Raises
    ------
    ValueError
        If the experiment kind or the device is not supported.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config_snapshot.json"
    kind = cfg["experiment"]["kind"]
    status = {
        "schema_version": 1,
        "run_id": cfg.get("run_id", run_dir.name),
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate", {"experiment": cfg["experiment"]["name"]}),
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(PROJECT_DIR),
        "dataset": _dataset_block(cfg),
        "selection": {
            "name": "root fidelity; no selection (fixed settings, all rounds kept)",
            "direction": "maximize",
            "split": "not applicable: one target per seed, no held-out data",
            "checkpoint": "final rounds (mean over evaluation.final_window)",
        },
        "log_path": "run.log",
        "config_path": config_path.name,
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    try:
        logger.info(
            "RUN_STARTED | run_id=%s | experiment=%s | kind=%s",
            status["run_id"],
            cfg["experiment"]["name"],
            kind,
        )
        if kind not in KINDS:
            raise ValueError(f"experiment.kind must be one of {KINDS}, got {kind!r}")
        if cfg["device"] != "cpu":
            raise ValueError(
                "This implementation runs on CPU only (numpy simulator, complex128 "
                f"torch ceiling); device must be 'cpu', got {cfg['device']!r}"
            )
        start = time.perf_counter()
        metrics = {"experiment": cfg["experiment"], "kind": kind}
        if kind == "shot_noise_tvd":
            logger.info("DATASET_READY | uniform 16-outcome distribution")
            metrics.update(_run_tvd(cfg))
        else:
            inputs = _inputs(cfg)
            identity = inputs_identity(inputs)
            status["dataset"]["sha256"] = identity
            write_json(run_dir / "run_status.json", status)
            logger.info(
                "DATASET_READY | targets=%s | stuck_per_seed=%d | sha256=%s",
                inputs.targets.shape,
                int(inputs.stuck_mask[0].sum()),
                identity,
            )
            metrics["seeds"] = inputs.seeds.tolist()
            metrics["stuck_per_seed"] = inputs.stuck_mask.sum(axis=(1, 2)).tolist()
            if kind == "qgan":
                metrics.update(_run_qgan(cfg, inputs, run_dir))
            else:
                metrics.update(_run_ceiling(cfg, inputs, run_dir))
        metrics["seconds"] = time.perf_counter() - start
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
            metrics["seconds"],
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
