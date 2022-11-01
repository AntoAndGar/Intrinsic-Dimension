from torchvision.datasets import MNIST, CIFAR10, FashionMNIST


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
        # set information on the dataset
        # MNIST dataset parameters
        CHANNEL_IN = 1
        INPUT_HEIGHT = 28
        INPUT_WIDTH = 28
        OUTPUT_DIM = 10
        VAL_GLOBAL_ACCURACY = 0.90
    elif dataset_name == "CIFAR10":
        # download and load CIFAR10 Dataset
        train_dataset = CIFAR10(
            root="./data/CIFAR10", download=True, train=True, transform=img_transform
        )
        test_dataset = CIFAR10(
            root="./data/CIFAR10", download=True, train=False, transform=img_transform
        )
        # set information on the dataset
        # CIFAR10 dataset parameters
        CHANNEL_IN = 3
        INPUT_HEIGHT = 32
        INPUT_WIDTH = 32
        OUTPUT_DIM = 10
        VAL_GLOBAL_ACCURACY = 0.46
    elif dataset_name == "FMNIST":
        # download and load Fashion MNIST Dataset
        train_dataset = FashionMNIST(
            root="./data/FashionMNIST", download=True, train=True, transform=img_transform
        )
        test_dataset = FashionMNIST(
            root="./data/FashionMNIST", download=True, train=False, transform=img_transform
        )
        # set information on the dataset
        # FMNIST dataset parameters
        CHANNEL_IN = 1
        INPUT_HEIGHT = 28
        INPUT_WIDTH = 28
        OUTPUT_DIM = 10
        VAL_GLOBAL_ACCURACY = 0.88
    else:
        raise Exception("Name of dataset not in: [MNIST, CIFAR10]")

    return (
        train_dataset,
        test_dataset,
        CHANNEL_IN,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        OUTPUT_DIM,
        VAL_GLOBAL_ACCURACY,
    )
