import random

import numpy as np
import torch

from aggregators import SignGuard
from attacks import a_little_is_enough_attack
from datasets import get_dataset
from experiments import evaluate_model, train_client
from models import get_model_for_dataset

# Determine the device to use (cuda, mps, or cpu)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

# Define training parameters
epochs = 20
epoch_milestones = [10, 15]
dataset = "mnist"
batch_size = 32
lr = 0.01
momentum = 0.9

dirichlet_alpha = 0.4

num_workers = 4
num_byzantines = 1

# Initialize model for the dataset
print(f"Initializing model for dataset: {dataset}")
model = get_model_for_dataset(dataset)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Load the dataset
train_loader, test_loader = get_dataset(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    dirichlet_alpha=dirichlet_alpha,
    seed=seed,
)
print(f"Dataset loaded: {len(train_loader[0].dataset)} training samples ({'non-iid' if dirichlet_alpha != 0 else 'iid'}), {len(test_loader.dataset)} test samples")  # noqa: E501

# Initialize the optimizer and scheduler
optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, milestones=epoch_milestones, gamma=0.1
)

# Initialize the loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Calculate the number of iterations per epoch
iterations_per_epoch = min(len(client_loader) for client_loader in train_loader)

print(
    f"""\nStarting training: {epochs} epochs, {num_workers} workers ({num_byzantines} Byzantine) and {iterations_per_epoch} iterations per epoch\n"""  # noqa: E501
)

byz_ratios = []
losses = []
model.to(device)
signguard = SignGuard(f=num_byzantines, use_dbscan=True)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for iteration in range(iterations_per_epoch):
        honest_gradients = []
        byz_gradients = []
        local_losses = []

        # Simulate byzantine workers
        for idx in range(num_byzantines):
            grad, loss = train_client(
                model, train_loader[idx], optimizer, criterion, device
            )
            byz_gradients.append(grad)

        # Simulate honest workers
        for idx in range(num_byzantines, num_workers):
            grad, loss = train_client(
                model, train_loader[idx], optimizer, criterion, device
            )
            honest_gradients.append(grad)
            local_losses.append(loss)

        # Perform the a-little-is-enough attack to generate byzantine gradients
        byz_gradients = a_little_is_enough_attack(byz_gradients, num_byzantines, z=0.5)

        # Combine byzantine and honest gradients
        gradients = byz_gradients + honest_gradients

        # Aggregate gradients
        aggregated_gradients, selected_idx, byz_ratio = signguard.aggregate(gradients)

        byz_ratios.append(byz_ratio)
        losses.append(sum(local_losses) / len(local_losses))

        # Update model parameters
        offset = 0
        for parameter in model.parameters():
            numel = parameter.numel()
            parameter.grad = (
                aggregated_gradients[offset : offset + numel]
                .reshape(parameter.shape)
                .clone()
            )
            offset += numel

        optimizer.step()

    # Evaluate model on test set
    test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f} %, Test Loss: {test_loss:.4f}")

    if scheduler is not None:
        scheduler.step()

print(f"Byzantine Ratio: {sum(byz_ratios) / len(byz_ratios):.2f}")
print(f"Loss: {sum(losses) / len(losses):.4f}")
