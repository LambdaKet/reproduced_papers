from __future__ import annotations

import json

import pytest
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_the_evidence_files(tmp_path):
    from DQuLearn.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["execution"]["backend"] = "local"
    cfg["training"]["circuits_per_epoch"] = 16
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    paper_runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert status["status"] == "COMPLETED"
    assert metrics["epochs"][0]["circuits"] == 16
