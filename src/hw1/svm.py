import logging

import torch
from torch import nn
from torch.optim.adam import Adam

from src.hw1.main import train_data, train_labels, test_data, test_labels


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
        scores = self.forward(x)
        # logger.debug()
        correct_scores = scores[[torch.arange(scores.size(0)), y]]
        margins = torch.clamp(scores - correct_scores[:, None] + 1, min=0)
        margins[torch.arange(scores.size(0)), y] = 0
        loss = torch.sum(margins)
        return loss

    def l2_regularization(self) -> torch.Tensor:
        return torch.sum(self.weights ** 2)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.hinge_loss(x, y) + self.l2_regularization()

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.state_dict(), path)


def train_svm(model: SVM, x: torch.Tensor, y: torch.Tensor, learning_rate: float = 1e-3, n_iters: int = 1000):
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
    model = train_svm(model, torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.uint8), n_iters=300)
    accuracy = test_svm(model, torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.uint8))
    print(f"Accuracy: {accuracy}")
    model.save("svm.pth")


if __name__ == "__main__":
    main()
