# Accuracy gap

Scope: partial: QuClassi's released preprocessing (random 1,000-image subsample, 4 PCA features, 5 qubits), lr 0.01 selected on the validation split of seed 0, 3 seeds. Accuracies in %, mean ± std over seeds 0, 1, 2.

Known-answer check, DQuLearn rule on 2 workers, pair 0/1: test 99.46 ± 0.03, train 99.2 ± 0.7.

Test accuracy of logistic regression on the same preprocessing, by number of PCA features (random 1,000-image subsample; last logistic column: every training image, 4 features; sweep `accuracy_feature_ceiling`), against the quantum model and the paper:

| Pair | 4 features | 6 features | 8 features | 16 features | 32 features | 64 features | 4 features, all images | DQuLearn rule, 4 features | Paper |
|---|---|---|---|---|---|---|---|---|---|
| 3/9 | 95.8 ± 0.4 | 95.4 ± 0.9 | 94.2 ± 1.1 | 95.6 ± 0.8 | 95.4 ± 0.1 | 95.8 ± 0.3 | 96.2 ± 0.1 | 95.2 ± 0.7 | 97.5 |
| 3/8 | 75.3 ± 1.1 | 82.9 ± 0.3 | 89.3 ± 1.5 | 92.0 ± 1.2 | 93.4 ± 1.6 | 93.3 ± 2.2 | 75.2 ± 0.1 | 72.6 ± 0.9 | 96.2 |
| 3/6 | 95.1 ± 0.2 | 96.7 ± 0.4 | 97.2 ± 0.6 | 97.5 ± 0.8 | 98.3 ± 0.4 | 98.3 ± 0.2 | 95.7 ± 0.0 | 95.8 ± 0.1 | 98.1 |
| 1/5 | 95.0 ± 2.9 | 94.2 ± 3.4 | 95.1 ± 2.5 | 96.5 ± 0.6 | 97.4 ± 0.8 | 97.2 ± 1.3 | 97.4 ± 0.0 | 93.7 ± 0.7 | 98.6 |

DQuLearn rule on 3/8 with 6 features (7 qubits): test 77.9 ± 0.4, train 79.0 ± 4.8.

DQuLearn rule with 4 features, train and test accuracy, and convergence (mean shifted loss averaged over seeds, change over the last five of 25 epochs):

| Pair | Train | Test | Loss at epoch 1, 5, 10, 15, 20, 25 | Change, epochs 20 → 25 |
|---|---|---|---|---|
| 3/9 | 94.6 ± 2.3 | 95.2 ± 0.7 | 1.370, 0.786, 0.719, 0.719, 0.719, 0.719 | +0.00% |
| 3/8 | 73.5 ± 2.7 | 72.6 ± 0.9 | 1.296, 0.889, 0.740, 0.738, 0.738, 0.738 | -0.00% |
| 3/6 | 96.1 ± 0.4 | 95.8 ± 0.1 | 1.290, 0.860, 0.731, 0.730, 0.730, 0.730 | -0.00% |
| 1/5 | 92.7 ± 1.4 | 93.7 ± 0.7 | 1.148, 0.681, 0.675, 0.675, 0.674, 0.674 | -0.01% |
