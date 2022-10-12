import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

# torch.autograd.set_detect_anomaly(True) #this line can have huge performance impact
# train the model

# training step
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    # train_loss_averager = make_averager()  # mantain a running average of the loss

    # TRAIN
    tqdm_iterator = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="",
        leave=True,
    )

    len_tr_dl_ds = len(train_loader.dataset)

    for batch_idx, (data, target) in tqdm_iterator:
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        output = model(data)
        loss = F.cross_entropy(output, target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # train_loss_averager(loss.item())
        tqdm_iterator.set_description(
            f"Train Epoch: {epoch} [ {batch_idx * len(data)}/{len_tr_dl_ds} \tLoss: {loss.item():.6f}]")
    tqdm_iterator.close()

    if np.isnan(loss.item()):
        print("Loss is nan")
        raise Exception("Loss is nan")
        exit()


# testing step
def test(model, test_loader, epoch, device):
    model.eval()
    test_loss = 0
    correct = 0

    tqdm_iterator = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="",
        leave=True,
    )

    len_ts_dl_ds = len(test_loader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm_iterator:
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            # get the index of the max probability
            pred = output.max(1, keepdim=True)[1] # equal to argmax
            correct += pred.eq(target.view_as(pred)).cpu().sum().item()
            tqdm_iterator.set_description(
                f"Test Epoch: {epoch} [{batch_idx * len(data)}/{len_ts_dl_ds} \tLoss: {test_loss:.6f}, Accuracy: {correct}/{len_ts_dl_ds} ({100.0 * correct / len_ts_dl_ds}%)"
            )
        test_loss /= len(test_loader.dataset)
        print(f"Validation Average loss: {test_loss:.6f}")

        tqdm_iterator.close()

    # show an histogram of the weights of the model
    """start = -1
    stop = 1
    bins = 30
    for param in model.parameters():
        if param.requires_grad:
            
            hist = torch.histc(param.data, bins = bins, min = start, max = stop)
            x = np.arange(start, stop, (stop-start)/bins)
            plt.bar(x, hist.cpu(), align='center')
            plt.ylabel('Frequency')
            plt.show() """

    return correct / len(test_loader.dataset), test_loss


def fit(
    args,
    model_intrinsic,
    device,
    train_dataloader,
    test_dataloader,
    optimizer,
    val_global_accuracy,
):
    best_acc, best_epoch = 0, 0
    for epoch in range(1, args.num_epochs + 1):
        train(model_intrinsic, train_dataloader, optimizer, epoch, device)
        accuracy, val_avg_loss = test(model_intrinsic, test_dataloader, epoch, device)
        print("Validation Accuracy: {}".format(accuracy))
        # save information to file
        with open(args.training_result_file, "a") as f:
            f.write(f"\nEpoch: {epoch}")
            f.write(f"\nValidation Average Loss: {val_avg_loss}")
            f.write(f"\nValidation Accuracy: {accuracy}")
        f.close()
        # save the model if it reach validation global accuracy > 90% of the actual SOTA model
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            if best_acc >= val_global_accuracy:
                saving_path = (
                    f"./{args.model_save_path}/{args.architecture}/{args.dataset}/"
                )
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                if args.architecture == "fcn":
                    torch.save(
                        model_intrinsic.state_dict(),
                        f"{saving_path}{args.architecture}_h{args.hidden_dim}_id{args.intrinsic_dim}_lay{args.num_layers}_lr{args.learning_rate}_proj_{args.projection}_opt_{args.optimizer}.pt",
                    )
                else:
                    torch.save(
                        model_intrinsic.state_dict(),
                        f"{saving_path}{args.architecture}_id{args.intrinsic_dim}_lr{args.learning_rate}_proj_{args.projection}_opt_{args.optimizer}.pt",
                    )

    return epoch, best_epoch, accuracy, best_acc
