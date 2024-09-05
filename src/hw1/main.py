from __future__ import annotations

import logging
import os.path
from typing import TYPE_CHECKING, Literal, cast, Any

import numpy as np
import xxhash
from annoy import AnnoyIndex
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.hw1 import unpickle_data

if TYPE_CHECKING:
    import numpy.typing as npt


N_NEIGHBORS = 3  # Experimentally determined to be the best number of neighbors

logger = logging.getLogger(__name__)

Metric = Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']

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
_test_data = unpickle_data('cifar-10-batches-py/test_batch')
test_data = _test_data["data"]
test_labels = _test_data["labels"]


def classify(X: npt.ArrayLike, p: int = 2) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Classify some data using the K-Nearest Neighbors algorithm, with all the training data."""

    classifier = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, p=p).fit(train_data, train_labels)
    prediction: np.ndarray[tuple[int], np.dtype[np.uint8]] = classifier.predict(X)
    return prediction


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
    def __init__(self, weights: Literal["uniform", "distance"] = "uniform", metric: Metric = "euclidean", num_trees: int = 20, n_neighbors: int = N_NEIGHBORS, save_index: bool = True):
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
