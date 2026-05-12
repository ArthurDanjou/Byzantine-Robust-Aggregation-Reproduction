import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def mnist():
    """Download and return the MNIST dataset.

    Returns:
    -------
        tuple: (train, test) datasets.
    """
    train = datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    test = datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    return train, test


def cifar10():
    """Download and return the CIFAR-10 dataset.

    Returns:
    -------
        tuple: (train, test) datasets.
    """
    train = datasets.CIFAR10(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    test = datasets.CIFAR10(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    return train, test


def get_dataset(
    name: str,
    num_workers: int = 1,
    batch_size: int = 32,
    dirichlet_alpha: float = 0,
) -> tuple[list[DataLoader], DataLoader]:
    """Return the dataset by name.

    Args:
    ----
        name (str): The name of the dataset to load.
        num_workers (int, optional): The number of federated clients to create.
            Defaults to 1.
        batch_size (int, optional): The batch size to use.
            Defaults to 32.
        dirichlet_alpha (float, optional): The alpha parameter for Dirichlet sampling.
            Defaults to 0.

    Returns:
    -------
        tuple[list[DataLoader], DataLoader]: The list of train data loaders
            and the test data loader.
    """
    if name == "mnist":
        train, test = mnist()
    elif name == "cifar10":
        train, test = cifar10()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    data_per_worker = len(train) // num_workers

    if dirichlet_alpha == 0:
        lengths = [data_per_worker] * num_workers
        lengths[-1] += len(train) - sum(lengths)

        iid_datasets = random_split(train, lengths)

        train_loader = [
            DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in iid_datasets
        ]
    elif dirichlet_alpha > 0:
        train_loader = sample_dirichlet_niid_loaders(
            train_dataset=train,
            num_workers=num_workers,
            alpha=dirichlet_alpha,
            batch_size=batch_size,
        )
    else:
        raise ValueError("alpha_iid must be greater or equal than 0")

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def sample_dirichlet_niid_loaders(
    train_dataset: torch.utils.data.Dataset,
    num_workers: int,
    alpha: float = 0.5,
    batch_size: int = 32,
) -> list[DataLoader]:
    """Sample a list of DataLoaders for a Dirichlet-NIID split of the given dataset.

    Args:
    ----
        train_dataset (torch.utils.data.Dataset): The training dataset to split.
        num_workers (int): The number of workers to split the dataset into.
        alpha (float, optional): The Dirichlet distribution parameter.
            Defaults to 0.5.
        batch_size (int, optional): The batch size for the DataLoaders.
            Defaults to 32.

    Returns:
    -------
        A list of DataLoaders for the Dirichlet-NIID split of the dataset.
    """
    targets = np.array(train_dataset.targets)
    num_classes = len(np.unique(targets))

    client_indices = {i: [] for i in range(num_workers)}

    for k in range(num_classes):
        class_idx = np.where(targets == k)[0]
        np.random.shuffle(class_idx)

        proportions = np.random.dirichlet(np.repeat(alpha, num_workers))
        splits = (proportions * len(class_idx)).astype(int)

        offset = 0
        for i in range(num_workers):
            end = offset + splits[i]
            if i == num_workers - 1:
                client_indices[i].extend(class_idx[offset:])
            else:
                client_indices[i].extend(class_idx[offset:end])
            offset = end

    train_loaders = []
    for i in range(num_workers):
        np.random.shuffle(client_indices[i])
        subset = Subset(train_dataset, client_indices[i])
        train_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    return train_loaders


if __name__ == "__main__":
    train, test = cifar10()
    print(train.data.shape)
    print(test.data.shape)

    train, test = mnist()
    print(train.data.shape)
    print(test.data.shape)

    cifar10_niid = get_dataset("cifar10", dirichlet_alpha=0.5)
    cifar10_iid = get_dataset("cifar10")

    print(cifar10_niid)
    print(cifar10_iid)
    print(cifar10_iid == cifar10_niid)
