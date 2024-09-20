from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hw1 import train_data, train_labels
from hw1.cross_validation import find_best_k_with_cross_validation
from hw1.knn import AnnoyClassifier

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def main():
    find_best_k_with_cross_validation(
        X=train_data,
        y=train_labels,
        classifier_cls=AnnoyClassifier,
        init_kwargs={
            "metric": "angular",
            "weights": "distance",
            "num_trees": 100,
            "save_index": False,
        },
    )


if __name__ == "__main__":
    main()
