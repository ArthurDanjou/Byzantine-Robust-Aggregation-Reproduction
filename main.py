import torch

from aggregators import MultiKrum
from attacks import fall_of_empires_attack
from datasets import get_dataset
from experiments import test_classification, train_step
from models import get_model_for_dataset

device = torch.device(
    "mps"
    if torch.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

epochs = 20
epoch_milestones = [10, 15]
dataset = "mnist"
batch_size = 32
lr = 0.01
momentum = 0.9
local_iter = 1

num_workers = 4
num_byzantines = 1


print(f"Initializing model for dataset: {dataset}")
model = get_model_for_dataset(dataset).to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
train_loader, test_loader = get_dataset(
    dataset, batch_size=batch_size, num_workers=num_workers
)

optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer, milestones=epoch_milestones, gamma=0.1
)
criterion = torch.nn.CrossEntropyLoss()

iteration_per_epoch = len(train_loader[0].dataset) // (batch_size * local_iter)

print(
    f"""\nStarting training: {epochs} epochs,
{num_workers} workers ({num_byzantines} Byzantine)"""
)
for epoch in range(epochs):
    print(f"\n{'=' * 40}\nEpoch {epoch + 1}/{epochs}\n{'=' * 40}")
    loss = train_step(
        model=model,
        aggregator=MultiKrum(
            n=num_workers, f=num_byzantines, m=num_workers - num_byzantines
        ),
        attack=fall_of_empires_attack,
        attack_kwargs={},
        device=device,
        iterations=iteration_per_epoch,
        num_byzantines=num_byzantines,
        num_workers=num_workers,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )
    accuracy = test_classification(
        device=device,
        model=model,
        test_loader=test_loader,
    )
    print(f"Test accuracy: {accuracy:.4f} at epoch {epoch + 1}")
