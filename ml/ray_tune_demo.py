"""
This script demonstrates hyperparameter tuning for a convolutional neural network (CNN) using the MNIST dataset.
It leverages Ray Tune for efficient hyperparameter optimization, specifically utilizing the Optuna search algorithm.
The script includes a CNN definition, data loading, training, and testing functions. The aim is to find the best
learning rate and momentum for the optimizer that maximizes the model's accuracy on the MNIST dataset.

To run this script, you need to install the following dependencies:
    - torch==2.0.0
    - torchvision==0.15.1
    - ray==2.8.0
    - optuna==3.4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray import train as ray_train


class ConvNet(nn.Module):
    """
    A simple convolutional neural network with two convolutional layers and two fully connected layers.
    Designed for classification tasks on the MNIST dataset.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Defines the forward pass of the CNN."""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_data():
    """
    Loads the MNIST dataset from torchvision datasets.
    Returns data loaders for both training and testing datasets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def train_model(model, optimizer, train_loader, device):
    """
    Trains the model for one epoch over the training dataset.
    Arguments:
        model: The neural network model to be trained.
        optimizer: The optimizer used for training.
        train_loader: DataLoader for the training dataset.
        device: The device (CPU/GPU) on which the training is executed.
    """
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test_model(model, test_loader, device):
    """
    Tests the model over the testing dataset.
    Arguments:
        model: The neural network model to be tested.
        test_loader: DataLoader for the testing dataset.
        device: The device (CPU/GPU) on which the testing is executed.
    Returns:
        accuracy: The computed accuracy of the model over the testing dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def objective(config):
    """
    The objective function for hyperparameter tuning with Ray Tune.
    Arguments:
        config: Configuration for hyperparameters to be tuned.
    Reports back the mean accuracy after training and testing the model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_data()
    model = ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=config["lr"], momentum=config["momentum"])

    for epoch in range(10):
        train_model(model, optimizer, train_loader, device)
        accuracy = test_model(model, test_loader, device)
        ray_train.report({"mean_accuracy": accuracy})


if __name__ == "__main__":
    # Hyperparameter search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "momentum": tune.uniform(0.1, 0.9)
    }

    # Optuna search algorithm
    algo = OptunaSearch()

    # Ray Tune configuration
    analysis = tune.run(
        objective,
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
        num_samples=10,
        config=search_space,
        stop={"training_iteration": 10},
    )

    # Getting the best model configuration
    best_config = analysis.get_best_config(metric="mean_accuracy", mode="max")
    print("Best config is:", best_config)
