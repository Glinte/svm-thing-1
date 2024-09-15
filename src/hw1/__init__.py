from __future__ import annotations

import pickle
from os import PathLike

from typing import TypedDict, Literal

import numpy as np


type N_IMAGES = int
type N_PIXELS = Literal[3072]
Data = np.ndarray[tuple[N_IMAGES, N_PIXELS], np.dtype[np.uint8]]  # Shape: (number_images, 32 * 32 * 3 = 3072)


class DataDict(TypedDict):
    batch_label: bytes
    labels: list[int]
    data: Data
    filenames: list[bytes]


def unpickle_data(fp: str | bytes | PathLike[str] | PathLike[bytes]) -> DataDict:
    """Unpickle the CIFAR-10 dataset."""
    with open(fp, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return {k.decode(): v for k, v in dict.items()}  # Convert bytes key to string key  # type: ignore
