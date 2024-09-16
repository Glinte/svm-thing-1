from __future__ import annotations

from typing import Literal, Any, cast

import numpy as np
import xxhash
from annoy import AnnoyIndex
from numpy import typing as npt
from sklearn.neighbors import KNeighborsClassifier

from hw1.main import N_NEIGHBORS, Metric
from hw1 import train_data, train_labels


def classify(X: npt.ArrayLike, p: int = 2) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Classify some data using the K-Nearest Neighbors algorithm, with all the training data."""

    classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, p=p).fit(train_data, train_labels)
    prediction: np.ndarray[tuple[int], np.dtype[np.uint8]] = classifier.predict(X)
    return prediction


class AnnoyClassifier:
    """K-Nearest Neighbors classifier using the Annoy library."""
    def __init__(self, weights: Literal["uniform", "distance"] = "uniform", metric: Metric = "euclidean", num_trees: int = 20, n_neighbors: int = N_NEIGHBORS, save_index: bool = False):
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
    save_index: bool = False,
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
