import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from sklearn import metrics
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hw1 import get_test_set_dataloader, label_names, get_train_set_dataloader

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self, channels: int) -> None:
        """Initialize the CNN.

        Args:
            channels: Number of channels in the input images.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels * 2, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_loop(
        self,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 2,
        device: torch.device = torch.device("cuda"),
        save_to: str | None = None,
    ) -> None:
        """Train the model."""
        self.to(device=device)
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate[tuple[torch.Tensor, torch.Tensor]](dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs.to(device=device))
                loss = criterion(outputs, labels.to(device=device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    logger.info(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0

        if save_to is not None:
            torch.save(self.state_dict(), save_to)
            logger.info(f"Saved model to {save_to}")


def main():
    logging.basicConfig(level=logging.INFO)

    additional_features: Any = []  # Any to avoid type checking error

    net = CNN(channels=3+len(additional_features))
    net.to(device=torch.device("cuda"))

    net.train_loop(
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(net.parameters(), lr=0.001),
        dataloader=get_train_set_dataloader(additional_features=additional_features),
        epochs=2,
        save_to=f"../../data/models/{datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M")}_cnn_Adam_2.pth",
        device=torch.device("cuda"),
    )
    # net.load_state_dict(torch.load("../../data/models/cnn_tutorial_2.pth", weights_only=True))

    y_pred = np.array([])
    y_true = np.array([])
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in get_test_set_dataloader(additional_features=additional_features):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.to(device=torch.device("cuda")))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
            y_true = np.concatenate((y_true, labels.numpy()))

    print(metrics.classification_report(y_true, y_pred, target_names=label_names, digits=4))


if __name__ == "__main__":
    main()
