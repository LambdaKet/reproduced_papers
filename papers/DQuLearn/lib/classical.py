"""Logistic-regression context baseline on the same features (not a claim of the paper).

Gives the accuracy a simple classical model reaches on the very features the
quantum model encodes (min-max scaled PCA components), for cross-paper context.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from .data import PairDataset


def logistic_regression_baseline(
    dataset: PairDataset, c_grid: list[float], seed: int
) -> dict:
    """Fit one logistic regression per C and select C on the validation split.

    Selection rule: highest validation accuracy; ties go to the smaller C.

    Parameters
    ----------
    dataset : PairDataset
        Splits of the digit pair; ``features`` are the model inputs.
    c_grid : list of float
        Inverse regularization strengths to try.
    seed : int
        Solver seed.

    Returns
    -------
    dict
        Every candidate's train and validation accuracy, the selected C, and
        its train, validation and test accuracy.

    Raises
    ------
    ValueError
        If the validation split is empty or the grid is empty.
    """
    if len(dataset.validation) == 0:
        raise ValueError("selecting C needs a non-empty validation split")
    if not c_grid:
        raise ValueError("classical.c_grid is empty")
    candidates, models = [], []
    for c_value in sorted(float(c) for c in c_grid):
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=seed)
        model.fit(dataset.train.features, dataset.train.labels)
        models.append(model)
        candidates.append(
            {
                "C": c_value,
                "train_accuracy": float(
                    model.score(dataset.train.features, dataset.train.labels)
                ),
                "validation_accuracy": float(
                    model.score(dataset.validation.features, dataset.validation.labels)
                ),
            }
        )
    # max() returns the first maximum, i.e. the smallest C among ties.
    best_index = max(
        range(len(candidates)), key=lambda i: candidates[i]["validation_accuracy"]
    )
    best, best_model = candidates[best_index], models[best_index]
    return {
        "candidates": candidates,
        "selection_rule": "highest validation accuracy; ties go to the smaller C",
        "selected_C": best["C"],
        "train_accuracy": best["train_accuracy"],
        "validation_accuracy": best["validation_accuracy"],
        "test_accuracy": float(
            best_model.score(dataset.test.features, dataset.test.labels)
        ),
    }
