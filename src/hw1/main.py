from __future__ import annotations

import os.path
from collections.abc import Sequence
from typing import TypedDict, TYPE_CHECKING, Literal, cast, Any
import pickle
import random

import numpy as np
import xxhash
from annoy import AnnoyIndex
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

if TYPE_CHECKING:
    from os import PathLike
    from PIL.Image import Image as ImageType
    import numpy.typing as npt


N_NEIGHBORS = 1  # Experimentally determined to be the best number of neighbors

Metric = Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']

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


def visualize_single_image(data: Sequence[bytes | int | float]) -> ImageType:
    """Visualize a single image."""

    image = Image.fromarray(np.array(data, dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0))
    image.show()
    return image


def classify(X: npt.ArrayLike, p: int = 2) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Classify some data using the K-Nearest Neighbors algorithm, with all the training data."""

    classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, p=p).fit(train_data, train_labels)
    prediction: np.ndarray[tuple[int], np.dtype[np.uint8]] = classifier.predict(X)
    return prediction


def find_best_k(
    X_train: npt.ArrayLike | None = None,
    y_train: npt.ArrayLike | None = None,
    X_test: npt.ArrayLike | None = None,
    y_test: npt.ArrayLike | None = None,
    classifier_cls: type[KNeighborsClassifier] | type[AnnoyClassifier] | None = None,
    init_kwargs: dict[str, Any] | None = None,
) -> int:
    """Find the best k for the K-Nearest Neighbors algorithm."""
    if classifier_cls is None:
        raise ValueError("classifier_cls must be provided.")

    if X_train is None:
        X_train = train_data
    if y_train is None:
        y_train = train_labels
    if X_test is None:
        X_test = test_data["data"]
    if y_test is None:
        y_test = test_data["labels"]

    best_k = 0
    best_score = 0
    for k in range(1, 10):
        if issubclass(classifier_cls, (KNeighborsClassifier, AnnoyClassifier)):
            classifier = classifier_cls(n_neighbors=k, **(init_kwargs or {}))
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
        else:
            raise ValueError("Invalid classifier.")
        score = metrics.f1_score(y_test, y_pred, average='macro')
        print(f"k: {k}, F1 score: {score}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best k: {best_k}, F1 score: {best_score}")
    return best_k


def find_best_k_with_cross_validation(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    classifier_cls: type[KNeighborsClassifier] | type[AnnoyClassifier] | None = None,
    init_kwargs: dict[str, Any] | None = None,
    n_splits: int = 5,
) -> int:
    """Find the best k for the K-Nearest Neighbors algorithm using cross-validation.

    Args:
        X: The data.
        y: The labels.
        classifier_cls: The classifier class.
        init_kwargs: The initialization keyword arguments.
        n_splits: The number of splits. This is the number of times the data is split into training and testing sets to do cross-validation.
    """
    if classifier_cls is None:
        raise ValueError("classifier_cls must be provided.")

    best_k = 0
    best_score = 0
    for k in range(1, 10):
        scores = []

        for i in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, train_size=0.9, shuffle=True, random_state=42+i)  # Fixed random state for reproducibility and caching
            if issubclass(classifier_cls, (KNeighborsClassifier, AnnoyClassifier)):
                classifier = classifier_cls(n_neighbors=k, **(init_kwargs or {}))
            else:
                raise ValueError("Invalid classifier.")
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            score = metrics.f1_score(y_test, y_pred, average='macro')
            print(f"k: {k}, test: {i+1}, F1 score: {score}")
            scores.append(score)

        avg_score = np.mean(scores)
        print(f"k: {k}, F1 score: {avg_score}")
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    print(f"Best k: {best_k}, F1 score: {best_score}")
    return best_k


class AnnoyClassifier:
    """K-Nearest Neighbors classifier using the Annoy library."""
    def __init__(self, weights: Literal["uniform", "distance"] = "uniform", metric: Metric = "euclidean", num_trees: int = 20, n_neighbors: int = 1, save_index: bool = True):
        self.weights = weights
        self.metric: Metric = metric
        self.num_trees = num_trees
        self.n_neighbors = n_neighbors
        self.index: AnnoyIndex | None = None
        self.labels: np.ndarray | None = None
        self.save_index = save_index

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> AnnoyClassifier:
        """Fit the model."""

        self.labels = np.array(y)
        self.index = build_index(X, metric=self.metric, num_trees=self.num_trees, save_index=self.save_index)
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Predict the labels of the data."""

        if self.index is None:
            raise ValueError("Index must be built first.")
        assert self.labels is not None

        X = np.asarray(X)
        include_distances = self.weights == "distance"
        y_pred = np.apply_along_axis(self._predict_single, 1, X, include_distances=include_distances)

        return y_pred

    def _predict_single(self, x: np.ndarray[tuple[int], Any], include_distances: bool) -> np.intp:
        """Predict the label of a single data point."""

        assert self.index is not None
        assert self.labels is not None

        if include_distances:
            include_distances = cast(Literal[True], include_distances)
            nns, distances = self.index.get_nns_by_vector(x, self.n_neighbors, include_distances=include_distances)
        else:
            include_distances = cast(Literal[False], include_distances)
            nns = self.index.get_nns_by_vector(x, self.n_neighbors, include_distances=include_distances)
        nn_labels = [self.labels[idx] for idx in nns]

        if include_distances:
            # Weighted voting
            weights = 1 / np.array(distances)  # pyright: ignore[reportPossiblyUnboundVariable]
            pred = np.argmax(np.bincount(nn_labels, weights=weights))
        else:
            pred = np.argmax(np.bincount(nn_labels))

        return pred


def build_index(
    data: npt.ArrayLike,
    metric: Metric = "euclidean",
    num_trees: int = 20,
    save_index: bool = True,
) -> AnnoyIndex:
    """Build an Annoy index if it doesn't exist, otherwise simply loads it."""

    index = AnnoyIndex(3072, metric)
    data_hash = xxhash.xxh64(data).hexdigest()  # type: ignore
    index_path = f'index_{metric}_{num_trees}_{data_hash}.ann'
    if os.path.exists(index_path):
        index.load(index_path)
    else:
        data = np.asarray(data)
        for i, vec in enumerate(data):
            index.add_item(i, vec)
        index.build(num_trees)
        if save_index:
            index.save(index_path)
    return index


def main():
    find_best_k_with_cross_validation(
        X=train_data,
        y=train_labels,
        classifier_cls=AnnoyClassifier,
        init_kwargs={"metric": "angular", "weights": "distance", "num_trees": 100, "save_index": False},
    )


if __name__ == "__main__":
    main()
