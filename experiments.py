import torch
from torch.utils.data import DataLoader


def train_client(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Train the client model on the given data using the specified optimizer.

    Args:
    ----
        model (nn.Module): The client model to train.
        train_loader (DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (torch.nn.Module): The loss function to use for training.
        device (torch.device): The device to use for training.

    Returns:
    -------
        tuple[torch.Tensor, float]: The concatenated gradients and the loss value.
    """
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    gradients = [
        param.grad.detach().reshape(-1).to(device)
        if param.grad is not None
        else torch.zeros_like(param).reshape(-1).to(device)
        for param in model.parameters()
    ]
    return torch.cat(gradients), loss.item()


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on the given test data.

    Args:
    ----
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The data loader for the test data.
        criterion (torch.nn.Module): The loss function to use for evaluation.
        device (torch.device): The device to use for evaluation.

    Returns:
    -------
        tuple[float, float]: The accuracy (%) and average loss on the test data.
    """
    model.eval()
    correct = 0
    total_loss = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    avg_loss = total_loss / total

    return acc, avg_loss
