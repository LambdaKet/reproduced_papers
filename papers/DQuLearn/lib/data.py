"""MNIST loading and QuClassi preprocessing.

Follows QuClassi's released code (``main.py`` lines 34–58, commit ``8e85e08``),
which DQuLearn's accuracy experiments repeat (DQuLearn Sec. IV-B):

1. scale pixels to [0, 1] and fit PCA on all 60,000 training images, keeping as
   many components as the circuit encodes (QuClassi: 4 for 5 qubits);
2. keep a subsample of the training images (QuClassi: the first 1,000);
3. min-max scale each component to [0, 1] on that subsample and encode it as
   the angle ``2·asin(√x)``;
4. keep the two digits of the binary task.

Added by this reproduction: a seeded validation split of the pair's subsample,
and the pair's test images, which go through the same PCA and scaling and are
clipped to [0, 1] (QuClassi's code has no test set). PCA uses the deterministic
``covariance_eigh`` solver; QuClassi's default solver is randomized and unseeded.

The data root comes from ``cfg["data_root"]`` (``--data-root``), else the
``DATA_DIR`` environment variable, else the repository's ``data/`` folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from runtime_lib.data_paths import resolve_data_root

PAPER_DATA_DIR = "DQuLearn"
MNIST_FILE = Path("MNIST") / "mnist.npz"
SUBSAMPLE_MODES = ("first", "random")
SPLITS = ("train", "validation", "test")
REACQUIRE_HINT = (
    "download it with: curl -fsSL -o data/DQuLearn/MNIST/mnist.npz "
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz "
    "(see the README, Data)"
)


@dataclass(frozen=True)
class SplitArrays:
    """One split of the binary task.

    Parameters
    ----------
    angles : numpy.ndarray
        Encoding angles, shape ``(n, n_features)``, in ``[0, π]``.
    features : numpy.ndarray
        Min-max scaled PCA features in ``[0, 1]`` (the classical baseline's input).
    labels : numpy.ndarray
        Class indices: 0 for the pair's first digit, 1 for the second.
    """

    angles: np.ndarray
    features: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class PairDataset:
    """Train, validation and test splits of one MNIST digit pair."""

    train: SplitArrays
    validation: SplitArrays
    test: SplitArrays
    meta: dict = field(default_factory=dict)

    def split(self, name: str) -> SplitArrays:
        if name not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {name!r}")
        return getattr(self, name)


def mnist_path(data_root: str | Path | None) -> Path:
    """Path of ``mnist.npz`` under the shared data root.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    """
    path = resolve_data_root(data_root) / PAPER_DATA_DIR / MNIST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"MNIST not found at {path}; {REACQUIRE_HINT}")
    return path


def _pair_from(value) -> tuple[int, int]:
    pair = tuple(int(digit) for digit in value)
    if len(pair) != 2 or pair[0] == pair[1] or not all(0 <= d <= 9 for d in pair):
        raise ValueError(f"data.pair must be two different digits, got {value!r}")
    return pair


def _encode(features: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.sqrt(features))


@lru_cache(maxsize=4)
def _load(path: str, n_features: int, data_key: str, seed: int) -> PairDataset:
    data_cfg = json.loads(data_key)
    pair = _pair_from(data_cfg["pair"])
    subsample = data_cfg["subsample"]
    mode, size = subsample["mode"], int(subsample["size"])
    validation_fraction = float(data_cfg["validation_fraction"])
    if mode not in SUBSAMPLE_MODES:
        raise ValueError(f"data.subsample.mode must be one of {SUBSAMPLE_MODES}")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be in [0, 1)")

    with np.load(path, allow_pickle=False) as archive:
        train_images = archive["x_train"].reshape(-1, 784) / 255.0
        train_digits = archive["y_train"].astype(int)
        test_images = archive["x_test"].reshape(-1, 784) / 255.0
        test_digits = archive["y_test"].astype(int)
    if not 0 < size <= len(train_digits):
        raise ValueError(f"data.subsample.size must be in 1..{len(train_digits)}")

    pca = PCA(n_components=n_features, svd_solver="covariance_eigh").fit(train_images)
    rng = np.random.default_rng(seed)
    if mode == "first":
        kept = np.arange(size)
    else:
        kept = np.sort(rng.choice(len(train_digits), size=size, replace=False))
    components = pca.transform(train_images[kept])
    lower = components.min(axis=0)
    span = components.max(axis=0) - lower
    span[span == 0.0] = 1.0

    def scale(values: np.ndarray) -> np.ndarray:
        return np.clip((values - lower) / span, 0.0, 1.0)

    def pair_split(features: np.ndarray, digits: np.ndarray) -> SplitArrays:
        in_pair = np.isin(digits, pair)
        selected = features[in_pair]
        labels = (digits[in_pair] == pair[1]).astype(int)
        return SplitArrays(_encode(selected), selected, labels)

    pool = pair_split(scale(components), train_digits[kept])
    order = rng.permutation(len(pool))
    n_validation = int(round(validation_fraction * len(pool)))
    validation_index = np.sort(order[:n_validation])
    train_index = np.sort(order[n_validation:])

    def take(split: SplitArrays, index: np.ndarray) -> SplitArrays:
        return SplitArrays(
            split.angles[index], split.features[index], split.labels[index]
        )

    test_components = pca.transform(test_images)
    test = pair_split(scale(test_components), test_digits)
    unclipped_test = (test_components[np.isin(test_digits, pair)] - lower) / span
    meta = {
        "name": "MNIST",
        "pair": list(pair),
        "n_features": n_features,
        "subsample": {"mode": mode, "size": size},
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "counts": {
            "train": int(len(train_index)),
            "validation": int(n_validation),
            "test": int(len(test)),
        },
        "test_values_clipped_fraction": float(
            np.mean((unclipped_test < 0.0) | (unclipped_test > 1.0))
        ),
    }
    return PairDataset(
        train=take(pool, train_index),
        validation=take(pool, validation_index),
        test=test,
        meta=meta,
    )


def prepare_pair_dataset(
    data_cfg: dict, n_features: int, seed: int, data_root: str | Path | None
) -> PairDataset:
    """Build the train, validation and test splits of one digit pair.

    Parameters
    ----------
    data_cfg : dict
        The config's ``data`` section: ``pair``, ``subsample`` (``mode``,
        ``size``) and ``validation_fraction``.
    n_features : int
        PCA components kept, the circuit's data features.
    seed : int
        Seed of the random subsample and of the validation split.
    data_root : str or pathlib.Path or None
        Shared data root (``cfg["data_root"]``).

    Returns
    -------
    PairDataset
        The three splits and a description in ``meta``.
    """
    key = json.dumps(
        {
            "pair": list(_pair_from(data_cfg["pair"])),
            "subsample": data_cfg["subsample"],
            "validation_fraction": data_cfg["validation_fraction"],
        },
        sort_keys=True,
    )
    return _load(str(mnist_path(data_root)), int(n_features), key, int(seed))


class ArrayLoader:
    """Iterates over a split in fixed order, in batches of angles and labels.

    Parameters
    ----------
    split : SplitArrays
        Data to iterate over.
    batch_size : int
        Samples per batch (QuClassi trains one sample at a time).
    """

    def __init__(self, split: SplitArrays, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.split = split
        self.batch_size = batch_size

    def __len__(self) -> int:
        return -(-len(self.split) // self.batch_size)

    def __iter__(self):
        for start in range(0, len(self.split), self.batch_size):
            stop = start + self.batch_size
            yield self.split.angles[start:stop], self.split.labels[start:stop]


def build_dataloader(cfg: dict, split: str) -> ArrayLoader:
    """Loader over one split of the configured digit pair.

    Parameters
    ----------
    cfg : dict
        Resolved run configuration (``data``, ``model.qubits``, ``seed``,
        ``data_root``).
    split : str
        ``train``, ``validation`` or ``test``.

    Returns
    -------
    ArrayLoader
        Batches of ``(angles, labels)`` with ``n_qubits − 1`` angles per sample.
    """
    dataset = prepare_pair_dataset(
        cfg["data"],
        n_features=int(cfg["model"]["qubits"]) - 1,
        seed=int(cfg["seed"]),
        data_root=cfg.get("data_root"),
    )
    return ArrayLoader(dataset.split(split), int(cfg["data"]["batch_size"]))
