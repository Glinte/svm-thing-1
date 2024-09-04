from __future__ import annotations

import pickle
from os import PathLike

from typing import TypedDict

import numpy as np


class Data(TypedDict):
    batch_label: bytes
    labels: list[int]
    data: np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # Shape: (number_images, 32 * 32 * 3 = 3072)
    filenames: list[bytes]


def unpickle_data(fp: str | bytes | PathLike[str] | PathLike[bytes]) -> Data:
    """Unpickle the CIFAR-10 dataset."""
    with open(fp, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return {k.decode(): v for k, v in dict.items()}  # Convert bytes key to string key  # type: ignore
