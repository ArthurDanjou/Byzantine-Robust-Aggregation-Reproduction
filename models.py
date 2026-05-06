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


class BasicBlock(nn.Module):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """Initialize the BasicBlock.

        Args:
        ----
            in_planes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the convolutional layers.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BasicBlock.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape
                (batch_size, in_planes, height, width).

        Returns:
        -------
            torch.Tensor: Output tensor of shape
                (batch_size, self.expansion * planes, height, width).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self, block, num_blocks, num_classes: int = 10) -> None:
        """Initialize the ResNet.

        Args:
        ----
            block (nn.Module): Block type to use (BasicBlock or Bottleneck).
            num_blocks (list): Number of blocks for each layer.
            num_classes (int): Number of output classes.
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride: int) -> nn.Sequential:
        """Create a layer of blocks.

        Args:
        ----
            block (nn.Module): Block type to use (BasicBlock or Bottleneck).
            planes (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the convolutional layers.

        Returns:
        -------
            nn.Sequential: Layer of blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    """Create a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2])


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
        return ResNet18()
    elif dataset_name == "mnist":
        return CNN()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
