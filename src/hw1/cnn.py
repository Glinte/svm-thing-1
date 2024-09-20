import logging

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hw1 import get_test_set_dataloader, label_names

logger = logging.getLogger(__name__)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
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


def train(net: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, dataloader: DataLoader[torch.Tensor], epochs: int = 2, device: torch.device = torch.device('cuda'), save_to: str | None = None) -> None:
    """Train the model."""
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs: torch.Tensor
            labels: torch.Tensor
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device=device))
            loss = criterion(outputs, labels.to(device=device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    if save_to is not None:
        torch.save(net.state_dict(), save_to)
        logger.info(f"Saved model to {save_to}")


def main():
    logging.basicConfig(level=logging.INFO)
    net = CNN()
    net.to(device=torch.device('cuda'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train(net, criterion, optimizer, get_train_set_dataloader(), epochs=5, save_to=f"../../data/models/{datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M")}_cnn_tutorial_2.pth", device=torch.device('cuda'))
    net.load_state_dict(torch.load("../../data/models/202409191050_cnn_tutorial_5.pth", weights_only=True))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in get_test_set_dataloader():
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.to(device=torch.device('cuda')))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(device=torch.device("cpu")) == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in label_names}
    total_pred = {classname: 0 for classname in label_names}

    # again no gradients needed
    with torch.no_grad():
        for data in get_test_set_dataloader():
            images, labels = data
            outputs = net(images.to(device=torch.device('cuda')))
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[label_names[label]] += 1
                total_pred[label_names[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    main()
