import argparse

""" BATCH_SIZE = 128  # Number of samples in each batch
DATASET_NAME = "MNIST"  # supported possibility: "MNIST" or "CIFAR10"
NETWORK_TYPE = "untied_lenet"  # supported possibility: "fcn", "lenet", "untied_lenet", "fc_lenet", "fc_tied_lenet"
HIDDEN_DIM = 200  # Number of neurons in each hidden layer for Fully Connected Network
NUM_LAYERS = 1  # Number of hidden layers for Fully Connected Network
INTRINSIC_DIM = 300  # Intrinsic dimension of the network
LEARNING_RATE = 0.01  # Learning rate for the optimizer
EPOCHS = 100  # Number of epochs to train the model 

TRAINING_RESULT_FILE = "results.txt"  # File to save the training results
MODEL_SAVE_PATH = "model/"  # Path to save the model """


def get_input_args():
    parser = argparse.ArgumentParser(
        description="Execute the test on the intrinsic dimension for various architecture, dimensionality, hyperparameter and dataset"
    )
    parser.add_argument(
        "-id",
        "--intrinsic_dim",
        type=int,
        help="an integer for the intrinsic dimension",
        required=True,
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        help="a string for the optimizer, can be in ['sgd', 'adam']",
        default="sgd",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="a float for the learning rate",
        required=True,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="an integer for the batch size",
        default=128,
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        help="a string for the dataset, can be in ['mnist', 'cifar10']",
        default="mnist",
    )
    parser.add_argument(
        "-l2",
        "--l2_reg",
        type=float,
        help="a float for the l2 regularization",
        default=0.0,
    )
    parser.add_argument(
        "-arch",
        "--architecture",
        type=str,
        help="a string for the architecture, can be in ['fcn', 'lenet', 'untied_lenet', 'fc_lenet', 'fc_tied_lenet', 'resnet']",
        required=True,
    )
    parser.add_argument(
        "-ne",
        "--num_epochs",
        type=int,
        help="an integer for the number of epochs",
        default=100,
    )
    parser.add_argument(
        "-hd",
        "--hidden_dim",
        type=int,
        help="an integer for the hidden dimension of the 'fcn' architecture",
        default=200,
    )
    parser.add_argument(
        "-nl",
        "--num_layers",
        type=int,
        help="an integer for the number of layers of the 'fcn' architecture",
        default=1,
    )
    parser.add_argument(
        "-msp",
        "--model_save_path",
        type=str,
        help="a string for the path (folders) to save the model",
        default="model",
    )
    parser.add_argument(
        "-trf",
        "--training_result_file",
        type=str,
        help="a string for the file to save the training results",
        default="results.txt",
    )
    parser.add_argument(
        "-p",
        "--projection",
        type=str,
        help="a string for the projection method, can be in [dense, fastfood]",
        default="dense",
    )
    parser.add_argument(
        "-mrp",
        "--model_result_path",
        type=str,
        help="a string for the path (folders) to save the model result file",
        default="results",
    )

    return parser.parse_args()
