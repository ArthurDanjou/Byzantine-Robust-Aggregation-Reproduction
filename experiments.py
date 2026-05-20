import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader


def train_client(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    return (
        parameters_to_vector(
            p.grad.detach() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ).to(device),
        loss.item(),
    )


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
