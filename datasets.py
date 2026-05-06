from torch.utils.data import DataLoader, Subset
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
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    return train, test


def get_dataset(
    name: str, num_workers: int = 1, batch_size: int = 32
) -> tuple[list[DataLoader], DataLoader]:
    """Return the dataset by name.

    Args:
    ----
        name (str): The name of the dataset to load.
        num_workers (int, optional): The number of worker threads to use. Defaults to 1.
        batch_size (int, optional): The batch size to use. Defaults to 32.

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
    train_loader = [
        DataLoader(
            Subset(train, range(i * data_per_worker, (i + 1) * data_per_worker)),
            batch_size=batch_size,
            shuffle=True,
        )
        for i in range(num_workers)
    ]

    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    train, test = cifar10()
    print(train.data.shape)
    print(test.data.shape)

    train, test = mnist()
    print(train.data.shape)
    print(test.data.shape)
