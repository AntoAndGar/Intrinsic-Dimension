from torchvision.datasets import MNIST, CIFAR10


def build_dataset(dataset_name, img_transform):
    train_dataset = None
    test_dataset = None

    if dataset_name == "MNIST":
        # download and load MNIST Dataset
        train_dataset = MNIST(
            root="./data/MNIST", download=True, train=True, transform=img_transform
        )
        test_dataset = MNIST(
            root="./data/MNIST", download=True, train=False, transform=img_transform
        )
    elif dataset_name == "CIFAR10":
        # download and load CIFAR10 Dataset
        train_dataset = CIFAR10(
            root="./data/CIFAR10", download=True, train=True, transform=img_transform
        )
        test_dataset = CIFAR10(
            root="./data/CIFAR10", download=True, train=False, transform=img_transform
        )
    else:
        raise Exception("Name of dataset not in: [MNIST, CIFAR10]")

    return train_dataset, test_dataset
