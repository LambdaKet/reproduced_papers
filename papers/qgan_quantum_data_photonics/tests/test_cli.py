from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_evidence(tmp_path):
    from qgan_quantum_data_photonics.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["training"]["rounds"] = 6
    cfg["evaluation"]["final_window"] = 3
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, default=str))

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert (run_dir / "trajectories.npz").exists()
    assert len(metrics["mean_fidelity_by_round"]) == 7
    assert metrics["final_fidelity"]["n"] == cfg["seeds"]["count"]
