import logging

import numpy as np
import torch
from torch import nn
from torch.optim.adam import Adam

from hw1 import (
    train_data as train_data_raw,
    train_labels,
    test_data as test_data_raw,
    test_labels,
    train_data_edges,
    test_data_edges,
)

logger = logging.getLogger(__name__)


class SVM(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super(SVM, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = nn.Parameter(torch.randn(n_features, n_classes))
        self.bias = nn.Parameter(torch.randn(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights) + self.bias

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)

    def hinge_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hinge loss of the SVM model."""
        scores = self.forward(x)
        correct_scores = scores[torch.arange(x.shape[0]), y]
        margins = torch.clamp(scores - correct_scores[:, None] + 1, min=0)
        margins[torch.arange(x.shape[0]), y] = 0
        return torch.mean(torch.sum(margins, dim=1))

    def l2_regularization(self) -> torch.Tensor:
        return torch.sum(self.weights**2)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.hinge_loss(x, y) + self.l2_regularization()

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def save(self, path: str):
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


def test_svm(model: SVM, x: torch.Tensor, y: torch.Tensor) -> float:
    y_pred = model.predict(x)
    accuracy = torch.mean((y_pred == y).float())
    return accuracy.item()


def main():
    n_features = 3072
    n_classes = 10

    model = SVM(n_features, n_classes)
    train_data = np.concatenate(
        (
            train_data_raw.reshape(50000, 3, 32, 32),
            train_data_edges.reshape(50000, 1, 32, 32),
        ),
        axis=1,
    )
    test_data = np.concatenate(
        (
            test_data_raw.reshape(10000, 3, 32, 32),
            test_data_edges.reshape(50000, 1, 32, 32),
        ),
        axis=1,
    )

    model = train_svm(
        model,
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.uint8),
        n_iters=50,
    )
    accuracy = test_svm(
        model,
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.uint8),
    )
    print(f"Accuracy: {accuracy}")
    model.save("svm.pth")


if __name__ == "__main__":
    main()
