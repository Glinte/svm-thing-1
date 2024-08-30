from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypedDict, TYPE_CHECKING
import pickle
import random

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

if TYPE_CHECKING:
    from os import PathLike
    from PIL.Image import Image as ImageType
    import numpy.typing as npt


N_NEIGHBORS = 1  # Experimentally determined to be the best number of neighbors

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


# From cifar-10-batches-py/batches.meta
batches_metadata = {'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_cases_per_batch': 10000, 'num_vis': 3072}
label_names = batches_metadata['label_names']

data_1 = unpickle_data('cifar-10-batches-py/data_batch_1')
data_2 = unpickle_data('cifar-10-batches-py/data_batch_2')
data_3 = unpickle_data('cifar-10-batches-py/data_batch_3')
data_4 = unpickle_data('cifar-10-batches-py/data_batch_4')
data_5 = unpickle_data('cifar-10-batches-py/data_batch_5')
train_data = np.concatenate((data_1["data"], data_2["data"], data_3["data"], data_4["data"], data_5["data"]))
train_labels = np.concatenate((data_1["labels"], data_2["labels"], data_3["labels"], data_4["labels"], data_5["labels"]))
test_data = unpickle_data('cifar-10-batches-py/test_batch')


def combine_images_horizontally(images: Sequence[ImageType]) -> ImageType:
    """Combine images horizontally."""
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.size[0]

    return new_image


def combine_images_vertically(images: Sequence[ImageType]) -> ImageType:
    """Combine images vertically."""
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)

    new_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    return new_image


def visualize_data_as_image(data: Data) -> ImageType:
    """Visualize the data as an image."""
    images = data["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    label_images = []
    for label_idx, _ in enumerate(label_names):
        # Choose 10 random images of the label
        images_of_label = [image for image, image_label in zip(images, data["labels"]) if image_label == label_idx]
        rand_10_images = [Image.fromarray(i) for i in random.choices(images_of_label, k=10)]
        label_images.append(combine_images_horizontally(rand_10_images))
    return combine_images_vertically(label_images)


def visualize_single_image(data: Sequence[bytes]) -> ImageType:
    """Visualize a single image."""
    image = Image.fromarray(np.array(data).reshape(3, 32, 32).transpose(1, 2, 0))
    image.show()
    return image


def classify(X: npt.ArrayLike, p: int = 2) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Classify some data using the K-Nearest Neighbors algorithm, with all the training data."""
    classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, p=p).fit(train_data, train_labels)
    prediction: np.ndarray[tuple[int], np.dtype[np.uint8]] = classifier.predict(X)
    return prediction


def find_best_k(
    train_X: npt.ArrayLike | None = None,
    train_y: npt.ArrayLike | None = None,
    test_X: npt.ArrayLike | None = None,
    test_y: npt.ArrayLike | None = None,
    metric: str = "minkowski",
    p: int = 2,
) -> int:
    if train_X is None:
        train_X = train_data
    if train_y is None:
        train_y = train_labels
    if test_X is None:
        test_X = test_data["data"]
    if test_y is None:
        test_y = test_data["labels"]

    best_k = 0
    best_score = 0
    for k in range(1, 10):
        classifier = KNeighborsClassifier(n_neighbors=k, p=p, metric=metric, n_jobs=-1).fit(train_X, train_y)
        score = metrics.f1_score(test_y, classifier.predict(test_X), average='macro')
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best k: {best_k}, F1 score: {best_score}")
    return best_k

def main():
    find_best_k(
        test_X=test_data["data"][0:1000],
        test_y=test_data["labels"][0:1000],
        p=1,
    )


if __name__ == "__main__":
    main()
