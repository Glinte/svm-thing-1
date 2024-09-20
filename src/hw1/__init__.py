from __future__ import annotations

import pickle
from os import PathLike

from typing import TypedDict, Literal

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

type N_IMAGES = int
type N_PIXELS = Literal[3072]
Data = np.ndarray[
    tuple[N_IMAGES, N_PIXELS], np.dtype[np.float64]
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
    (data_1["data"], data_2["data"], data_3["data"], data_4["data"], data_5["data"]), dtype=np.float64
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
test_data: Data = _test_data["data"].astype(np.float64)
test_data_edges: np.ndarray[
    tuple[N_IMAGES, Literal[32], Literal[32]], np.dtype[np.bool_]
] = np.load("test_data_edges.npy")
test_labels = _test_data["labels"]


def get_train_set_dataloader(batch_size: int = 16) -> DataLoader[torch.Tensor]:
    """Get the CIFAR-10 training DataLoader."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="../../data", train=True, download=False, transform=transform
    )
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


def get_test_set_dataloader(batch_size: int = 16) -> DataLoader[torch.Tensor]:
    """Get the CIFAR-10 test DataLoader."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="../../data", train=False, download=False, transform=transform
    )
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
