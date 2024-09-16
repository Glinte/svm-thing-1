from __future__ import annotations

import pickle
from os import PathLike

from typing import TypedDict, Literal

import numpy as np


type N_IMAGES = int
type N_PIXELS = Literal[3072]
Data = np.ndarray[
    tuple[N_IMAGES, N_PIXELS], np.dtype[np.uint8]
]  # Shape: (number_images, 32 * 32 * 3 = 3072)


class DataDict(TypedDict):
    batch_label: bytes
    labels: list[int]
    data: Data
    filenames: list[bytes]


def unpickle_data(fp: str | bytes | PathLike[str] | PathLike[bytes]) -> DataDict:
    """Unpickle the CIFAR-10 dataset."""
    with open(fp, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return {
        k.decode(): v for k, v in dict.items()
    }  # Convert bytes key to string key  # type: ignore


# From cifar-10-batches-py/batches.meta
batches_metadata = {
    "label_names": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "num_cases_per_batch": 10000,
    "num_vis": 3072,
}
label_names = batches_metadata["label_names"]
data_1 = unpickle_data("../../data/cifar-10-batches-py/data_batch_1")
data_2 = unpickle_data("../../data/cifar-10-batches-py/data_batch_2")
data_3 = unpickle_data("../../data/cifar-10-batches-py/data_batch_3")
data_4 = unpickle_data("../../data/cifar-10-batches-py/data_batch_4")
data_5 = unpickle_data("../../data/cifar-10-batches-py/data_batch_5")
train_data: Data = np.concatenate(
    (data_1["data"], data_2["data"], data_3["data"], data_4["data"], data_5["data"])
)
train_data_edges: np.ndarray[
    tuple[N_IMAGES, Literal[32], Literal[32]], np.dtype[np.bool_]
] = np.load("train_data_edges.npy")
train_labels = np.concatenate(
    (
        data_1["labels"],
        data_2["labels"],
        data_3["labels"],
        data_4["labels"],
        data_5["labels"],
    )
)
_test_data = unpickle_data("../../data/cifar-10-batches-py/test_batch")
test_data: Data = _test_data["data"]
test_data_edges: np.ndarray[
    tuple[N_IMAGES, Literal[32], Literal[32]], np.dtype[np.bool_]
] = np.load("test_data_edges.npy")
test_labels = _test_data["labels"]
