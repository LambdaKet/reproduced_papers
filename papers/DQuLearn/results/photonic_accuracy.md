# MerLin adaptation: accuracy parity with a photonic workload

Scope: partial, simulation only: MerLin adaptation of claim accuracy-parity; 2 photons in 4 modes (counterpart of QuClassi's 5-qubit circuit), exact probabilities, QuClassi's preprocessing (4 PCA features), learning rate per rule selected on the validation split of seed 0, 3 seeds.

Test accuracy (%), mean ± std over 3 seeds. Photonic model: 2 photons, 4 modes, input [1, 0, 1, 0], dual rail: per photon BS, PS(2 asin sqrt x1), BS, PS(2 asin sqrt x2); rectangular Mach-Zehnder mesh, 12 phases; MeasurementStrategy.probs(FOCK); probability of the input state; detector threshold (outcome |s>: one click on each occupied input mode; equal to its FOCK probability with no loss); postselection none; MerLin CPU simulator, exact probabilities (merlinquantum 0.4.2, perceval-quandela 1.3.1). Learning rates: DQuLearn rule, 2 workers 0.01, QuClassi rule, one process 0.01, exact photonic shift, 2 workers 0.01.

| Pair | Photonic, DQuLearn rule, 2 workers | Photonic, QuClassi rule, one process | Photonic, exact photonic shift, 2 workers | Qubit model, DQuLearn rule (Phase 03) | Paper |
|---|---|---|---|---|---|
| 3/9 | 82.8 ± 22.1 | 86.9 ± 8.9 | 81.1 ± 25.0 | 95.2 ± 0.7 | 97.5 |
| 3/8 | 66.0 ± 12.1 | 60.9 ± 11.5 | 65.9 ± 12.2 | 72.6 ± 0.9 | 96.2 |
| 3/6 | 89.1 ± 11.5 | 81.7 ± 21.3 | 90.9 ± 8.5 | 95.8 ± 0.1 | 98.1 |
| 1/5 | 89.9 ± 6.1 | 63.6 ± 22.5 | 83.0 ± 18.3 | 93.7 ± 0.7 | 98.6 |

Parity (rule fixed before the runs: |DQuLearn rule − QuClassi rule| ≤ 2 points on every pair, and identical parameters for local, 1-worker and 2-worker training): not kept.

| Pair | DQuLearn − QuClassi rule (points) | Photonic / qubit, DQuLearn rule (paired ratio [95% interval]) | Exact shift / two-point (paired ratio) |
|---|---|---|---|
| 3/9 | -4.06 | 0.869 [0.304, 1.435] (no difference) | 0.970 [0.843, 1.096] (no difference) |
| 3/8 | +5.11 | 0.908 [0.516, 1.299] (no difference) | 0.998 [0.987, 1.009] (no difference) |
| 3/6 | +7.40 | 0.930 [0.631, 1.230] (no difference) | 1.023 [0.925, 1.122] (no difference) |
| 1/5 | +26.30 | 0.960 [0.784, 1.137] (no difference) | 0.917 [0.553, 1.281] (no difference) |

Parity runs (3/6, seed 0, DQuLearn rule): largest parameter difference against local training: local 0, manager_w1 0, manager_w2 0; test accuracy local 95.63%, manager_w1 95.63%, manager_w2 95.63%.

Circuits per sample (each one a task for a worker): DQuLearn rule, 2 workers 24, QuClassi rule, one process 24, exact photonic shift, 2 workers 49. A difference is |mean ratio − 1| > 2 std of the paired ratios.

Gradients: parameter-shift estimates from exact probabilities; the two-point ±π/2 estimate is not the derivative for 2 photons, the exact shift rule is. Seeds: 0, 1, 2. Mean wall-clock per run (25 epochs and evaluations, this laptop's CPU): DQuLearn rule, 2 workers 38 s, QuClassi rule, one process 26 s, exact photonic shift, 2 workers 63 s. Simulated, not hardware-executed; no loss or noise.
