# Accuracy

Scope: partial: QuClassi's released preprocessing (random 1,000-image subsample, 4 PCA features, 5 qubits), lr 0.01 selected on the validation split of seed 0, 3 seeds. Test accuracy (%), mean ± std over seeds 0, 1, 2.

| Pair | DQuLearn rule, 2 workers | QuClassi rule, local | Logistic regression | DQuLearn / QuClassi (95% interval) | Paper, DQuLearn | QuClassi Fig. 9 |
|---|---|---|---|---|---|---|
| 3/9 | 95.2 ± 0.7 | 94.9 ± 0.7 | 95.8 ± 0.4 | 1.002 [1.002, 1.003] | 97.5 | 97.6 |
| 3/8 | 72.6 ± 0.9 | 72.1 ± 1.2 | 75.3 ± 1.1 | 1.008 [0.992, 1.023] | 96.2 | 87.0 |
| 3/6 | 95.8 ± 0.1 | 95.8 ± 0.1 | 95.1 ± 0.2 | 0.999 [0.997, 1.002] | 98.1 | 97.6 |
| 1/5 | 93.7 ± 0.7 | 93.9 ± 0.5 | 95.0 ± 2.9 | 0.998 [0.993, 1.003] | 98.6 | 96.8 |

Learning rate selected on validation (mean over the four pairs, seed 0): dqulearn: {'lr': 0.01} (0.0001: 0.404, 0.001: 0.648, 0.01: 0.921); quclassi: {'lr': 0.01} (0.0001: 0.560, 0.001: 0.560, 0.01: 0.907).

Parity (3/6, seed 0, lr 0.01): largest final-parameter difference to local training: manager_w1 0, manager_w2 0 (bit-identical).

Shot noise (3/6, seed 0, 2 workers, preliminary): test 95.48% with 10,000 shots vs 95.68% exact; largest parameter difference 0.0022 rad.
