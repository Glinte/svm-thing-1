from __future__ import annotations

import pickle
from os import PathLike
from pathlib import Path

from typing import TypedDict, Literal, Callable, Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

type N_IMAGES = int
type N_PIXELS = Literal[3072]
Data = np.ndarray[tuple[N_IMAGES, N_PIXELS], np.dtype[np.float64]]  # Shape: (number_images, 32 * 32 * 3 = 3072)


class DataDict(TypedDict):
    batch_label: bytes
    labels: list[int]
    data: Data
    filenames: list[bytes]


def unpickle_data(fp: str | bytes | PathLike[str] | PathLike[bytes]) -> DataDict:
    """Unpickle the CIFAR-10 dataset."""
    with open(fp, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return {k.decode(): v for k, v in dict.items()}  # Convert bytes key to string key  # type: ignore


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
train_data_edges: np.ndarray[tuple[N_IMAGES, Literal[32], Literal[32]], np.dtype[np.bool_]] = np.load(
    "train_data_edges.npy"
)
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
test_data_edges: np.ndarray[tuple[N_IMAGES, Literal[32], Literal[32]], np.dtype[np.bool_]] = np.load(
    "test_data_edges.npy"
)
test_labels = _test_data["labels"]


class CustomCIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR-10 dataset with additional custom features."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
        additional_features: list[Literal["edges", "corners"]] | None = None,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.additional_features = additional_features
        if additional_features is not None:
            if "edges" in additional_features:
                edges = train_data_edges if train else test_data_edges
                self.data = np.concatenate((self.data, edges.reshape(-1, 1, 32, 32).transpose(0, 2, 3, 1)), axis=3)
            if "corners" in additional_features:
                raise NotImplementedError("Corners feature is not implemented yet.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_train_set_dataloader(
    batch_size: int = 16, additional_features: list[Literal["edges", "corners"]] | None = None
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Get the CIFAR-10 training DataLoader."""
    channels = 3 + len(additional_features or [])
    normalize = transforms.Normalize((0.5,) * channels, (0.5,) * channels)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    trainset = CustomCIFAR10(
        root="../../data", train=True, download=False, transform=transform, additional_features=additional_features
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
    testset = CustomCIFAR10(root="../../data", train=False, download=False, transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def main():
    train_loader = get_train_set_dataloader(batch_size=16, additional_features=["edges"])
    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        break


if __name__ == "__main__":
    main()
    main()
