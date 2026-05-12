import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """A simple CNN model for MNIST classification."""

    def __init__(self):
        """Initialize the CNN model."""
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN model.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x: torch.Tensor) -> int:
        """Compute the number of flat features in the input tensor.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape
                (batch_size, channels, height, width).

        Returns:
        -------
            int: Number of flat features.
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CifarCNN(nn.Module):
    """CNN model for CIFAR-10 classification."""

    def __init__(self):
        """Initialize the CifarCNN model."""
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CifarCNN model.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x: torch.Tensor) -> int:
        """Compute the number of flat features in the input tensor.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape
                (batch_size, channels, height, width).

        Returns:
        -------
            int: Number of flat features.
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_model_for_dataset(dataset_name: str) -> nn.Module:
    """Get the model for the given dataset.

    Args:
    ----
        dataset_name (str): Name of the dataset.

    Returns:
    -------
        nn.Module: Model for the given dataset.
    """
    if dataset_name == "cifar10":
        return CifarCNN()
    elif dataset_name == "mnist":
        return CNN()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
