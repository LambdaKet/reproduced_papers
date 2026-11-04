from __future__ import annotations

import numpy as np
import pytest
from common import load_runtime_ready_config
from DQuLearn.lib.data import build_dataloader, mnist_path, prepare_pair_dataset


def _mnist_available() -> bool:
    try:
        mnist_path(None)
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _mnist_available(),
    reason="MNIST is not in data/DQuLearn/MNIST (see the README, Data)",
)


def test_loader_batches_have_one_angle_per_feature():
    cfg = load_runtime_ready_config()
    cfg["data"]["batch_size"] = 4
    angles, labels = next(iter(build_dataloader(cfg, "train")))
    assert angles.shape == (4, cfg["model"]["qubits"] - 1)
    assert np.all((angles >= 0) & (angles <= np.pi))
    assert set(labels) <= {0, 1}


def test_first_1000_images_hold_the_pair_counts_of_the_data_log():
    # Data Acquisition Log: 3/6 -> 187 and 1/5 -> 208 among the first 1,000 images.
    for pair, expected in (([3, 6], 187), ([1, 5], 208)):
        data_cfg = {
            "pair": pair,
            "subsample": {"mode": "first", "size": 1000},
            "validation_fraction": 0.2,
        }
        dataset = prepare_pair_dataset(data_cfg, n_features=4, seed=0, data_root=None)
        assert len(dataset.train) + len(dataset.validation) == expected
        assert len(dataset.validation) == round(0.2 * expected)


def test_labels_follow_the_pair_order():
    data_cfg = {
        "pair": [1, 5],
        "subsample": {"mode": "first", "size": 1000},
        "validation_fraction": 0.0,
    }
    dataset = prepare_pair_dataset(data_cfg, n_features=6, seed=0, data_root=None)
    assert dataset.train.angles.shape[1] == 6
    # 1/5 among the first 1,000 images: 116 ones and 92 fives in MNIST's order.
    assert np.bincount(dataset.train.labels).tolist() == [116, 92]
