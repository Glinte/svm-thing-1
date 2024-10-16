import logging
from typing import Literal

import numpy as np
import torch
from sklearn import metrics
from sklearn.decomposition import PCA
from torch import nn
from torch.optim.adam import Adam

from hw1 import (
    train_data as train_data_raw,
    train_labels,
    test_data as test_data_raw,
    test_labels,
    train_data_edges,
    test_data_edges,
    label_names,
)


logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SVM(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super(SVM, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = nn.Parameter(torch.randn(n_features, n_classes, device=device))
        self.bias = nn.Parameter(torch.randn(n_classes, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the SVM model."""
        return torch.matmul(x, self.weights) + self.bias

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the class of the input."""
        return torch.argmax(self.forward(x), dim=1)

    def hinge_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hinge loss of the SVM model."""
        scores = self.forward(x)
        correct_scores = scores[torch.arange(x.shape[0]), y]
        margins = torch.clamp(scores - correct_scores[:, None] + 1, min=0)
        margins[torch.arange(x.shape[0]), y] = 0
        return torch.mean(torch.sum(margins, dim=1))

    def l2_regularization(self) -> torch.Tensor:
        """Compute the L2 regularization term."""
        return torch.sum(self.weights**2)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the SVM model."""
        return self.hinge_loss(x, y) + self.l2_regularization()

    def load(self, path: str):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path, weights_only=True))

    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)


def train_svm(
    model: SVM,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    n_iters: int = 1000,
):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

    return model


def main():
    n_features = 32 * 32
    n_classes = 10
    n_iters = 20000

    model = SVM(n_features, n_classes)
    train_data = PCA(n_components=50).fit_transform(np.concatenate(
        (
            # train_data_raw.reshape(50000, 3, 32, 32),
            train_data_edges.reshape(50000, 1, 32, 32),
        ),
        axis=1,
    ))
    test_data = PCA(n_components=50).fit_transform(np.concatenate(
        (
            # test_data_raw.reshape(10000, 3, 32, 32),
            test_data_edges.reshape(10000, 1, 32, 32),
        ),
        axis=1,
    ))

    model = train_svm(
        model,
        torch.tensor(train_data.reshape(50000, -1), dtype=torch.float32, device=device),
        torch.tensor(train_labels, dtype=torch.int64, device=device),
        n_iters=n_iters,
    )
    # model.load(f"../../data/models/svm_edges_20000.pth")

    y_pred = model.predict(torch.tensor(test_data.reshape(10000, -1), dtype=torch.float32, device=device)).cpu().numpy()
    print(metrics.classification_report(test_labels, y_pred, target_names=label_names, digits=4))
    model.save(f"../../data/models/svm_edges_only_{n_iters}.pth")


if __name__ == "__main__":
    main()
