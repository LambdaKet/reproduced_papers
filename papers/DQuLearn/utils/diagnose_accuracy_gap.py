"""Diagnose the accuracy gap: known answer, feature count, train vs test, convergence.

Usage (from the repository root)::

    python papers/DQuLearn/utils/diagnose_accuracy_gap.py \\
        --sweeps papers/DQuLearn/outdir --results papers/DQuLearn/results

Reads three sweeps: ``sweep_accuracy_final`` (the quantum runs, with their
training histories), ``sweep_accuracy_diagnostics`` (the known-answer pair and
3/8 with 6 features) and ``sweep_accuracy_feature_ceiling`` (logistic regression
on the same preprocessing with 4 to 64 PCA features, and with every training
image at 4 features). Writes ``results/accuracy_gap.json`` and
``results/accuracy_gap.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from build_results import ACCURACY_SCOPE, PAPER_ACCURACY, _pm, _stat

PAIRS = ("3/9", "3/8", "3/6", "1/5")
FEATURES = (4, 6, 8, 16, 32, 64)
CHECKPOINTS = (1, 5, 10, 15, 20, 25)


def _summary(sweeps: Path, sweep: str) -> dict:
    summary = json.loads((sweeps / f"sweep_{sweep}" / "sweep_summary.json").read_text())
    if not summary["complete_sweep"]:
        raise SystemExit(f"{sweep} is not a complete sweep; refusing to use it")
    return summary


def _run_metrics(sweeps: Path, sweep: str, candidate: str) -> list[dict]:
    sweep_dir = sweeps / f"sweep_{sweep}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs = [r for r in status["runs"] if r["candidate"]["name"] == candidate]
    return [json.loads((sweep_dir / r["metrics_path"]).read_text()) for r in runs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sweeps", type=Path, required=True, help="Folder holding sweep_<id>/"
    )
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    final = _summary(args.sweeps, "accuracy_final")
    diagnostics = _summary(args.sweeps, "accuracy_diagnostics")
    ceiling = _summary(args.sweeps, "accuracy_feature_ceiling")

    data: dict = {"scope": ACCURACY_SCOPE, "known_answer": {}, "pairs": {}}
    known = {
        split: _stat(diagnostics, "dqulearn_pair01", f"{split}_accuracy")
        for split in ("train", "validation", "test")
    }
    data["known_answer"] = known
    six_features = {
        split: _stat(diagnostics, "dqulearn_q7_pair38", f"{split}_accuracy")
        for split in ("train", "validation", "test")
    }
    data["six_features_3_8"] = six_features

    for pair in PAIRS:
        key = pair.replace("/", "")
        quantum = {
            split: _stat(final, f"dqulearn_pair{key}", f"{split}_accuracy")
            for split in ("train", "validation", "test")
        }
        histories = [
            m["history"]
            for m in _run_metrics(args.sweeps, "accuracy_final", f"dqulearn_pair{key}")
        ]
        loss = {
            str(epoch): float(
                np.mean([h[epoch - 1]["mean_shifted_loss"] for h in histories])
            )
            for epoch in CHECKPOINTS
        }
        data["pairs"][pair] = {
            "logistic_by_features": {
                str(k): _stat(ceiling, f"logistic_f{k}_pair{key}", "test_accuracy")
                for k in FEATURES
            },
            "logistic_4_features_all_images": _stat(
                ceiling, f"logistic_f4_all_images_pair{key}", "test_accuracy"
            ),
            "dqulearn_rule_4_features": quantum,
            "mean_shifted_loss_by_epoch": loss,
            "loss_change_last_5_epochs": loss["25"] / loss["20"] - 1,
            "paper": {
                "dqulearn": PAPER_ACCURACY[pair][0],
                "quclassi_fig9": PAPER_ACCURACY[pair][1],
            },
        }

    lines = [
        "# Accuracy gap",
        "",
        f"Scope: {ACCURACY_SCOPE}. Accuracies in %, mean ± std over seeds 0, 1, 2.",
        "",
        f"Known-answer check, DQuLearn rule on 2 workers, pair 0/1: test {_pm(known['test'], 100, 2)}, "
        f"train {_pm(known['train'], 100, 1)}.",
        "",
        "Test accuracy of logistic regression on the same preprocessing, by number of PCA features "
        "(random 1,000-image subsample; last logistic column: every training image, 4 features; "
        "sweep `accuracy_feature_ceiling`), against the quantum model and the paper:",
        "",
        "| Pair | "
        + " | ".join(f"{k} features" for k in FEATURES)
        + " | 4 features, all images | DQuLearn rule, 4 features | Paper |",
        "|---|" + "---|" * (len(FEATURES) + 3),
    ]
    for pair, entry in data["pairs"].items():
        cells = [_pm(entry["logistic_by_features"][str(k)], 100, 1) for k in FEATURES]
        lines.append(
            f"| {pair} | "
            + " | ".join(cells)
            + f" | {_pm(entry['logistic_4_features_all_images'], 100, 1)}"
            + f" | {_pm(entry['dqulearn_rule_4_features']['test'], 100, 1)} | {entry['paper']['dqulearn']} |"
        )
    lines += [
        "",
        f"DQuLearn rule on 3/8 with 6 features (7 qubits): test {_pm(six_features['test'], 100, 1)}, "
        f"train {_pm(six_features['train'], 100, 1)}.",
        "",
        "DQuLearn rule with 4 features, train and test accuracy, and convergence (mean shifted "
        "loss averaged over seeds, change over the last five of 25 epochs):",
        "",
        "| Pair | Train | Test | Loss at epoch "
        + ", ".join(str(e) for e in CHECKPOINTS)
        + " | Change, epochs 20 → 25 |",
        "|---|---|---|---|---|",
    ]
    for pair, entry in data["pairs"].items():
        quantum = entry["dqulearn_rule_4_features"]
        losses = ", ".join(
            f"{v:.3f}" for v in entry["mean_shifted_loss_by_epoch"].values()
        )
        lines.append(
            f"| {pair} | {_pm(quantum['train'], 100, 1)} | {_pm(quantum['test'], 100, 1)} | {losses} | "
            f"{100 * entry['loss_change_last_5_epochs']:+.2f}% |"
        )
    data["sources"] = {
        sweep: f"outdir/sweep_{sweep}/sweep_summary.json"
        for sweep in (
            "accuracy_final",
            "accuracy_diagnostics",
            "accuracy_feature_ceiling",
        )
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "accuracy_gap.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.results / "accuracy_gap.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.results / 'accuracy_gap'}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
