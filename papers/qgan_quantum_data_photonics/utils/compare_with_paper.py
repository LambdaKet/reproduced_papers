"""Compare the validated sweeps with the paper and apply the predeclared rules.

Usage (from the repository root)::

    python papers/qgan_quantum_data_photonics/utils/compare_with_paper.py \\
        papers/qgan_quantum_data_photonics/outdir \\
        --out papers/qgan_quantum_data_photonics/results

Reads only the ``metrics.json`` files referenced by each ``sweep_status.json``
under ``outdir/sweep_*``. Writes ``comparison_with_paper.json`` and
``comparison_with_paper.md`` to ``--out`` and prints the Markdown. Paired
differences use the same target seeds in both arms (the inputs of a seed do not
depend on the condition); their 95 % intervals are percentile bootstraps over
seeds (10,000 resamples, generator seed 0). Decision rules, set before the
runs:

- phase noise: quantitatively reproduced if every level is within 0.03 of the
  paper; trend reproduced if the mean falls with sigma and crosses 0.90
  between 0.03 pi and 0.05 pi; unsupported if the mean is below 0.90 at some
  sigma <= 0.03 pi;
- defects: supported if the mean is at least 0.90 up to 14/30 stuck;
  unsupported if it is below 0.90 at 10/30 or fewer; the decline is called a
  property of the circuit only if the QGAN stays within 0.02 of the ceiling
  at every rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import paper_values as paper

RESAMPLES = 10_000


def _metrics(outdir: Path, sweep: str) -> dict[str, dict]:
    sweep_dir = outdir / f"sweep_{sweep}"
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep} is {status['status']}, not COMPLETED")
    return {
        run["run_id"]: json.loads((sweep_dir / run["metrics_path"]).read_text())
        for run in status["runs"]
    }


def _paired(a: list[float], b: list[float]) -> dict:
    """Mean of a - b over paired seeds with a 95 % percentile bootstrap interval."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, diff.size, (RESAMPLES, diff.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(diff.mean()),
        "ci95": [float(low), float(high)],
        "n": int(diff.size),
    }


def phase_noise(outdir: Path) -> dict:
    runs = _metrics(outdir, "phase_noise_sweep")
    variants = _metrics(outdir, "phase_noise_variants")
    ceiling = _metrics(outdir, "oracle_ceiling")["ceiling_stuck_0_of_30"]
    rows = []
    for sigma, paper_final in paper.PHASE_NOISE_FINAL.items():
        m = runs[f"sigma_{sigma}pi"]
        rows.append(
            {
                "sigma_pi": sigma,
                "final_mean": m["final_fidelity"]["mean"],
                "final_std": m["final_fidelity"]["std"],
                "noise_averaged_mean": m["noise_averaged_final_fidelity"]["mean"],
                "round_to_0p9": m["mean_fidelity_first_reaches"]["round"],
                "paper_final": paper_final,
                "paper_round_to_0p9": paper.PHASE_NOISE_ROUND_TO_0P9[sigma],
                "difference_to_paper": m["final_fidelity"]["mean"] - paper_final,
                "exact_tau_final_mean": variants[f"exact_tau_sigma_{sigma}pi"][
                    "final_fidelity"
                ]["mean"],
                "ceiling_noise_averaged": ceiling["noise_averaged_best_fidelity"][
                    str(sigma)
                ]["mean"],
            }
        )
    finals = [r["final_mean"] for r in rows]
    monotone = all(x >= y for x, y in zip(finals, finals[1:], strict=False))
    crosses = any(
        r["final_mean"] < 0.90 for r in rows if r["sigma_pi"] >= 0.04
    ) and all(r["final_mean"] >= 0.90 for r in rows if r["sigma_pi"] <= 0.03)
    quantitative = all(abs(r["difference_to_paper"]) <= 0.03 for r in rows)
    unsupported = any(r["final_mean"] < 0.90 for r in rows if r["sigma_pi"] <= 0.03)
    verdict = (
        "quantitatively reproduced"
        if quantitative
        else "trend reproduced"
        if monotone and crosses
        else "unsupported"
        if unsupported
        else "partially supported"
    )
    return {
        "rows": rows,
        "rules": {
            "every level within 0.03 of the paper": quantitative,
            "mean falls monotonically with sigma": monotone,
            "mean crosses 0.90 between 0.03 pi and 0.05 pi": crosses,
            "mean below 0.90 at some sigma <= 0.03 pi": unsupported,
        },
        "verdict": verdict,
    }


def defect_rate(outdir: Path) -> dict:
    runs = {}
    for part in ("low", "mid", "high"):
        runs.update(_metrics(outdir, f"defect_rate_sweep_{part}"))
    ceiling = _metrics(outdir, "oracle_ceiling")
    rows = []
    for stuck, paper_fidelity, paper_rounds in zip(
        paper.DEFECT_STUCK, paper.DEFECT_FIDELITY, paper.DEFECT_ROUNDS, strict=True
    ):
        m = runs[f"stuck_{stuck}_of_30"]
        c = ceiling[f"ceiling_stuck_{stuck}_of_30"]
        if m["seeds"] != c["seeds"] or m["stuck_per_seed"] != c["stuck_per_seed"]:
            raise ValueError(f"QGAN and ceiling runs are not paired at {stuck}/30")
        rows.append(
            {
                "stuck_of_30": stuck,
                "final_mean": m["final_fidelity"]["mean"],
                "final_std": m["final_fidelity"]["std"],
                "share_of_targets_at_least_0p9": float(
                    np.mean(np.asarray(m["final_fidelity_per_seed"]) >= 0.9)
                ),
                "paper_fidelity": paper_fidelity,
                "difference_to_paper": m["final_fidelity"]["mean"] - paper_fidelity,
                "convergence_round_mean": m["convergence_round"]["mean"],
                "paper_convergence_round": paper_rounds,
                "ceiling_mean": c["best_fidelity"]["mean"],
                "qgan_minus_ceiling": _paired(
                    m["final_fidelity_per_seed"], c["best_fidelity_per_seed"]
                ),
            }
        )
    by_stuck = {r["stuck_of_30"]: r for r in rows}
    supported = all(by_stuck[k]["final_mean"] >= 0.90 for k in range(0, 15, 2))
    unsupported = any(by_stuck[k]["final_mean"] < 0.90 for k in range(0, 11, 2))
    finals = [r["final_mean"] for r in rows]
    decreasing = finals[-1] < finals[0] and all(
        y <= x + 0.01 for x, y in zip(finals, finals[1:], strict=False)
    )
    within_ceiling = all(abs(r["qgan_minus_ceiling"]["mean"]) <= 0.02 for r in rows)
    return {
        "rows": rows,
        "rules": {
            "mean >= 0.90 up to 14/30 stuck": supported,
            "mean < 0.90 at 10/30 or fewer": unsupported,
            "mean decreases with the defect rate (steps of at most +0.01)": decreasing,
            "QGAN within 0.02 of the ceiling at every rate": within_ceiling,
        },
        "verdict": "supported" if supported and not unsupported else "unsupported",
        "mechanism": "circuit redundancy (QGAN at the ceiling)"
        if within_ceiling
        else "learning-limited at high defect rates (QGAN below the ceiling)",
    }


def shot_noise(outdir: Path) -> dict:
    runs = _metrics(outdir, "shot_noise_sweep")
    rows = [
        {
            "counts": counts,
            "final_mean": runs[f"counts_{counts}"]["final_fidelity"]["mean"],
            "final_std": runs[f"counts_{counts}"]["final_fidelity"]["std"],
            "paper_fidelity": paper_fidelity,
            "convergence_round_mean": runs[f"counts_{counts}"]["convergence_round"][
                "mean"
            ],
            "paper_convergence_round": paper_rounds,
        }
        for counts, paper_fidelity, paper_rounds in zip(
            paper.SHOT_COUNTS, paper.SHOT_FIDELITY, paper.SHOT_ROUNDS, strict=True
        )
    ]
    tvd_dir = outdir / "shot_noise_tvd_original"
    tvd_run = sorted(tvd_dir.glob("run_*"))[-1]
    tvd = json.loads((tvd_run / "metrics.json").read_text())["mean_tvd_by_count_level"]
    return {
        "rows": rows,
        "tvd": {
            k: {"ours": tvd[str(k)], "paper": v} for k, v in paper.TVD_ENDS.items()
        },
        "tvd_run": tvd_run.name,
    }


def chip_and_single(outdir: Path) -> dict:
    chip = _metrics(outdir, "chip_emulations")
    single = _metrics(outdir, "single_runs")
    noise = _metrics(outdir, "phase_noise_sweep")
    intact = {s: chip[f"intact_chip_emulation_sigma_{s}pi"] for s in (0.01, 0.02)}
    defective = {s: chip[f"defective_chip_emulation_sigma_{s}pi"] for s in (0.01, 0.02)}
    return {
        "intact_chip": {
            str(s): m["final_fidelity"] | {"paper_chip_run": paper.CHIP_INTACT_FIDELITY}
            for s, m in intact.items()
        },
        "defective_chip": {
            str(s): m["final_fidelity"]
            | {"paper_chip_run": paper.CHIP_DEFECTIVE_FIDELITY}
            for s, m in defective.items()
        },
        "entangled_state_learning_rule": {
            "mean >= 0.90 within 150 rounds at sigma 0.01 pi or 0.02 pi": any(
                m["mean_fidelity_by_round"][150] >= 0.90 for m in intact.values()
            )
        },
        "single_defect": single["single_defect"]["final_fidelity"]
        | {"paper": paper.SINGLE_DEFECT, "paper_no_defect": paper.NO_DEFECT},
        "fixed_discriminator_minus_qgan_at_0p02pi": _paired(
            single["fixed_discriminator"]["final_fidelity_per_seed"],
            noise["sigma_0.02pi"]["final_fidelity_per_seed"],
        ),
    }


def markdown(result: dict) -> str:
    pn, dr, sn, cs = (
        result["phase_noise"],
        result["defect_rate"],
        result["shot_noise"],
        result["chip_and_single"],
    )
    lines = [
        "### Phase noise (Fig. 2c): final root fidelity, 100 targets, 1500 counts",
        "",
        "| σ | Ours (mean ± std) | Paper (read) | Ours − paper | Round to 0.9 (ours / paper) | Exact tr[Mτ] | Ours, noise-averaged | Ceiling, noise-averaged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pn["rows"]:
        lines.append(
            f"| {r['sigma_pi']}π | {r['final_mean']:.3f} ± {r['final_std']:.3f} | {r['paper_final']:.2f} "
            f"| {r['difference_to_paper']:+.3f} | {r['round_to_0p9']} / {r['paper_round_to_0p9']} "
            f"| {r['exact_tau_final_mean']:.3f} | {r['noise_averaged_mean']:.3f} | {r['ceiling_noise_averaged']:.3f} |"
        )
    lines += ["", f"Rules: {pn['rules']}. Verdict: **{pn['verdict']}**.", ""]
    lines += [
        "### Defect rate (Fig. 3b): final root fidelity, 100 targets, σ = 0.02π, 1500 counts",
        "",
        "| Stuck / 30 | Ours (mean ± std) | Targets ≥ 0.9 | Paper (read) | Ours − paper | Ceiling | QGAN − ceiling [95 % CI] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in dr["rows"]:
        g = r["qgan_minus_ceiling"]
        lines.append(
            f"| {r['stuck_of_30']} | {r['final_mean']:.3f} ± {r['final_std']:.3f} "
            f"| {r['share_of_targets_at_least_0p9']:.0%} | {r['paper_fidelity']:.3f} "
            f"| {r['difference_to_paper']:+.3f} | {r['ceiling_mean']:.3f} "
            f"| {g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |"
        )
    lines += [
        "",
        f"Rules: {dr['rules']}. Verdict: **{dr['verdict']}**; mechanism: {dr['mechanism']}.",
        "",
        "### Shot noise (Suppl. Fig. 9): final root fidelity, 100 targets, σ = 0",
        "",
        "| Counts | Ours (mean ± std) | Paper (read) |",
        "|---:|---:|---:|",
    ]
    for r in sn["rows"]:
        lines.append(
            f"| {r['counts']} | {r['final_mean']:.3f} ± {r['final_std']:.3f} | {r['paper_fidelity']:.3f} |"
        )
    tvd = sn["tvd"]
    lines += [
        "",
        f"TVD of a uniform 16-outcome distribution: {tvd[100]['ours']:.4f} at 100 counts "
        f"(paper ≈{tvd[100]['paper']}), {tvd[10000]['ours']:.4f} at 10,000 (paper ≈{tvd[10000]['paper']}).",
        "",
        "### Chip runs replayed in simulation, single defect, fixed discriminator",
        "",
    ]
    for name in ("intact_chip", "defective_chip"):
        for s, v in cs[name].items():
            lines.append(
                f"- {name.replace('_', ' ')}, σ = {s}π: {v['mean']:.3f} ± {v['std']:.3f} "
                f"(20 targets; paper chip run {v['paper_chip_run']:.4f})"
            )
    sd = cs["single_defect"]
    fd = cs["fixed_discriminator_minus_qgan_at_0p02pi"]
    lines += [
        f"- Rule for the intact chip: {cs['entangled_state_learning_rule']}.",
        f"- One stuck phase shifter per circuit: {sd['mean']:.3f} ± {sd['std']:.3f} (paper {sd['paper'][0]:.4f} ± {sd['paper'][1]:.4f}; "
        f"no defect {sd['paper_no_defect'][0]:.4f} ± {sd['paper_no_defect'][1]:.4f}).",
        f"- Fixed discriminator minus QGAN at σ = 0.02π (paired, {fd['n']} targets): {fd['mean']:+.3f} "
        f"[{fd['ci95'][0]:+.3f}, {fd['ci95'][1]:+.3f}].",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "phase_noise": phase_noise(args.outdir),
        "defect_rate": defect_rate(args.outdir),
        "shot_noise": shot_noise(args.outdir),
        "chip_and_single": chip_and_single(args.outdir),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison_with_paper.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    text = markdown(result)
    (args.out / "comparison_with_paper.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
