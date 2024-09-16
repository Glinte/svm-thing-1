from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Any

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from hw1.knn import AnnoyClassifier
from src.hw1 import unpickle_data, Data

if TYPE_CHECKING:
    import numpy.typing as npt


N_NEIGHBORS = 3  # Experimentally determined to be the best number of neighbors

logger = logging.getLogger(__name__)

Metric = Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']

# From cifar-10-batches-py/batches.meta
batches_metadata = {'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_cases_per_batch': 10000, 'num_vis': 3072}
label_names = batches_metadata['label_names']

data_1 = unpickle_data('../../data/cifar-10-batches-py/data_batch_1')
data_2 = unpickle_data('../../data/cifar-10-batches-py/data_batch_2')
data_3 = unpickle_data('../../data/cifar-10-batches-py/data_batch_3')
data_4 = unpickle_data('../../data/cifar-10-batches-py/data_batch_4')
data_5 = unpickle_data('../../data/cifar-10-batches-py/data_batch_5')
train_data: Data = np.concatenate((data_1["data"], data_2["data"], data_3["data"], data_4["data"], data_5["data"]))
train_labels = np.concatenate((data_1["labels"], data_2["labels"], data_3["labels"], data_4["labels"], data_5["labels"]))
_test_data = unpickle_data('../../data/cifar-10-batches-py/test_batch')
test_data: Data = _test_data["data"]
test_labels = _test_data["labels"]


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
            logger.info(f"k: {k}, test: {i+1}, F1 score: {score}")
            scores.append(score)

        avg_score = np.mean(scores)
        print(f"k: {k}, F1 score: {avg_score}")
        if avg_score > best_score:
            best_score = avg_score
            best_k = k
    print(f"Best k: {best_k}, F1 score: {best_score}")
    return best_k


def main():
    find_best_k_with_cross_validation(
        X=train_data,
        y=train_labels,
        classifier_cls=AnnoyClassifier,
        init_kwargs={"metric": "angular", "weights": "distance", "num_trees": 100, "save_index": False},
    )


if __name__ == "__main__":
    main()
