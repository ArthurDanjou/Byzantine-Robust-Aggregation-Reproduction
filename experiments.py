from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from aggregators import Aggregator


def test_classification(
    device: torch.device, model: nn.Module, test_loader: DataLoader
) -> float:
    """Test the classification performance of a model on a given test loader.

    Args:
    ----
        device (torch.device): The device to run the test on.
        model (nn.Module): The model to test.
        test_loader (torch.utils.data.DataLoader): The test data loader.

    Returns:
    -------
        float: The accuracy of the model on the test set.

    """
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    print(f"Evaluating on {total} test samples...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / total
    print("Test Accuracy: %.2f %%" % (acc))
    return acc


def worker(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """
    Perform a training step on the model using the worker's data.

    Args:
    ----
        model (nn.Module): The model to train.
        train_loader (DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss function to use.
        device (torch.device): The device to use for training.

    Returns:
    -------
        tuple[torch.Tensor, float]: The gradients and loss value.
    """
    model.train()
    images, labels = next(train_loader)
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    grad = get_gradient_values(model)

    return grad, loss.item()


def get_gradient_values(model: nn.Module) -> torch.Tensor:
    """Get a reference to the flat gradient of the model.

    Args:
    ----
        model (nn.Module): The model to get the gradient of.

    Returns:
    -------
        torch.Tensor: The flat gradient of the model.
    """
    gradient = (
        torch.cat([torch.reshape(param.grad, (-1,)) for param in model.parameters()])
        .clone()
        .detach()
    )
    return gradient


def set_gradient_values(model: nn.Module, gradient: torch.Tensor) -> None:
    """Overwrite the gradient with the given one.

    Args:
    ----
        model (nn.Module): The model to set the gradient of.
        gradient (torch.Tensor): The flat gradient to set.
    """
    cur_pos = 0
    for param in model.parameters():
        param.grad = (
            torch.reshape(
                torch.narrow(gradient, 0, cur_pos, param.nelement()), param.size()
            )
            .clone()
            .detach()
        )
        cur_pos = cur_pos + param.nelement()


def train_step(
    model: nn.Module,
    num_workers: int,
    num_byzantines: int,
    train_loader: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    iterations: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    attack: Callable,
    attack_kwargs: dict,
    aggregator: Aggregator,
    criterion: nn.Module,
) -> tuple[list[float], list[float]]:
    """Train the model using the given data loader and optimizer.

    Args:
    ----
        model (nn.Module): The model to train.
        num_workers (int): The number of worker threads to use.
        num_byzantines (int): The number of byzantine workers to use.
        train_loader (list[DataLoader]): The list of data loaders for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        iterations (int): The number of iterations to train for.
        scheduler (LRScheduler): The learning rate scheduler to use.
        device (torch.device): The device to use for training.
        attack (Callable): The attack function to use.
        attack_kwargs (dict): Keyword arguments for the attack function.
        aggregator (Callable): The aggregator function to use.
        criterion (nn.Module): The loss function to use.

    Returns:
    -------
        tuple[list[float], list[float]]: list of loss for each iterations
            and list of byzantine rates
    """
    iter_loss = []
    byzantine_rates = []
    data_loader = []

    for idx in range(num_workers):
        data_loader.append(iter(train_loader[idx]))

    print(f"Starting training loop: {iterations} iterations")
    for it in range(iterations):
        local_losses = []
        honest_gradients = []
        byzantine_gradients = []

        # Honest workers [f -> n)
        for idx in range(num_byzantines, num_workers):
            grad, loss = worker(
                model, data_loader[idx], optimizer, criterion, device
            )
            honest_gradients.append(grad)
            local_losses.append(loss)

        # Byzantine workers [0 -> f)
        for idx in range(num_byzantines):
            grad, loss = worker(
                model, data_loader[idx], optimizer, criterion, device
            )
            byzantine_gradients.append(grad)

        iter_loss.append(local_losses)

        byzantine_gradients = attack(
            honest_gradients=honest_gradients,
            num_byzantine=num_byzantines,
            defense_func=aggregator,
            **attack_kwargs,
        )
        gradients = byzantine_gradients + honest_gradients
        gradients, selected_idx, byzantine_ratio = aggregator.aggregate(
            gradients=gradients
        )
        byzantine_rates.append(byzantine_ratio)
        set_gradient_values(model, gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_avg = sum(local_losses) / len(local_losses)
        iter_loss.append(loss_avg)

        if (it + 1) % 10 == 0:
            print(
                f"""Iteration {it + 1}/{iterations}, Loss: {loss_avg:.4f},
Byzantine ratio: {byzantine_ratio:.2f}"""
            )

    print(f"Training complete. Final loss: {iter_loss[-1]:.4f}")

    if scheduler is not None:
        scheduler.step()

    return iter_loss, byzantine_rates
