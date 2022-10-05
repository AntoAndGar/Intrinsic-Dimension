import torch
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import json
import os

from train import fit
from intrinsic_wrappers import *
from networks_arch import *
from utils import get_input_args
from model_builder import *
from dataset_builder import build_dataset


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU is not available")

    # add reproducibility stuff
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_input_args()

    # load the dataset and all the cooresponding parameters
    (
        train_dataset,
        test_dataset,
        CHANNEL_IN,
        INPUT_DIM,
        OUTPUT_DIM,
        VAL_GLOBAL_ACCURACY,
    ) = build_dataset(
        dataset_name=args.dataset.upper(),
        img_transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Create the file for logging training result, if it does not exist
    if not os.path.exists(args.training_result_file):
        open(args.training_result_file, "w").close()

    # load the model
    model, num_params = build_model(
        args.architecture,
        INPUT_DIM,
        args.hidden_dim,
        OUTPUT_DIM,
        args.num_layers,
        CHANNEL_IN,
        args.training_result_file,
        args.dataset,
        device,
    )

    # it is a little bit ugly but 0 hidden dimension means that we do not want to use the intrinsic dimension method to train our network and we use a normal training
    if args.intrinsic_dim > 0:
        # project the model on the subspace of dimension equal to intrinsic dimension
        if args.projection == "dense":
            # dense projection
            model_intrinsic = DenseWrap(model, args.intrinsic_dim, device)
        elif args.projection == "fastfood":
            # fastfood projection
            model_intrinsic = FastfoodWrapper(model, args.intrinsic_dim, device)
            # TODO: add the other projection methods like sparse and others
        else:
            raise Exception("Name of projection not in: [dense, fastfood]")
    else:
        # standard training
        model_intrinsic = model

    count_params(model=model_intrinsic, msg="Number of intrinsic parameters: ")
    with open(args.training_result_file, "a") as f:
        f.write(f"\nintrinsic_dim: {args.intrinsic_dim}")
    f.close()

    # load the optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model_intrinsic.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model_intrinsic.parameters(), lr=args.learning_rate)
    else:
        raise Exception("Name of optimizer not in: [sgd, adam]")

    # the training and testing dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # on Winzoz system this is needed, if you don't want wait forever for the creation of the workers at each epoch
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,  # on Winzoz system this is needed, if you don't want wait forever for the creation of the workers at each epoch
    )

    model_intrinsic.to(device)

    # train the model
    epoch, best_epoch, accuracy, best_acc = fit(
        args,
        model_intrinsic,
        device,
        train_dataloader,
        test_dataloader,
        optimizer,
        VAL_GLOBAL_ACCURACY,
    )

    # Create the path to save the results of training
    result_path = f"./{args.model_result_path}/{args.architecture}/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Create the file to store results, if it does not exist
    json_file = f"{result_path}results_{args.architecture}_{args.dataset}.json"
    if not os.path.exists(json_file):
        f = open(json_file, "w")
        f.write("{}")
        f.close()

    # save the results of training
    j = None
    with open(json_file, "r") as f:
        j = json.load(f)
    f.close()
    with open(json_file, "w") as f:
        if args.architecture == "fcn":
            j[
                f"{args.architecture}_model_h{args.hidden_dim}_id{args.intrinsic_dim}_lay{args.num_layers}_lr{args.learning_rate}_proj_{args.projection}_opt{args.optimizer}"
            ] = {
                "number_parameter": num_params,
                "hidden_dimension": args.hidden_dim,
                "number_layers": args.num_layers,
                "intrinsic_dimension": args.intrinsic_dim,
                "epoch": epoch,
                "validation_accuracy": accuracy,
                "learning_rate": args.learning_rate,
                "best_epoch": best_epoch,
                "best_accuracy": best_acc,
            }
        else:
            j[
                f"{args.architecture}_model_id{args.intrinsic_dim}_lr{args.learning_rate}_proj_{args.projection}_opt{args.optimizer}"
            ] = {
                "number_parameter": num_params,
                "intrinsic_dimension": args.intrinsic_dim,
                "epoch": epoch,
                "validation_accuracy": accuracy,
                "learning_rate": args.learning_rate,
                "best_epoch": best_epoch,
                "best_accuracy": best_acc,
            }
        json.dump(j, f, indent=4, separators=(",", ": "))
    f.close()
