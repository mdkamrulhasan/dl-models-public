import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from scipy.stats import loguniform
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import pickle
from pathlib import Path
from ray.train import Checkpoint, get_checkpoint
from ray import train
from ray import tune
import tempfile
import copy
from models import Net
import os


def training_hpo(config, data_dir=None, n_epoch=2):
    """helper function for hyperparamter optimization."""

    # Splitting training data into training and validation sets
    VALID_RATIO = 0.2
    vaildationset = copy.deepcopy(trainset)
    vaildationset.data = trainset.data[int(1 - VALID_RATIO * len(trainset.data)):]
    trainset.data = trainset.data[:int(1 - VALID_RATIO * len(trainset.data))]

    # Redefining training and validation data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(vaildationset, batch_size=config["batch_size"],
                                              shuffle=True, num_workers=2)

    # CNN model instantiation
    net = Net()
    net.to(device)

    # Define a Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Checkpoint related
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_loss = []
    # Model/network training
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # to GPU
            if device != 'cpu':
                inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print statistics (every 2000 mini-batches)
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        # logging training loss (KH)
        train_loss.append(loss.item())  ## may be running_loss more appropriate??

    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(validloader, 0):
        with torch.no_grad():
            inputs, labels = data
            if device != 'cpu':
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    checkpoint_data = {
        "epoch": n_epoch,
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)

        print("accuracy: ", correct / total)
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )


def optimize_hyperparams():
    max_num_epochs = 10
    num_samples = 10  # 50

    hp_config = {
        # "h1": tune.choice([2 ** i for i in np.arange(4, 9)]),
        # "h2": tune.choice([2 ** i for i in np.arange(4, 9)]),
        "lr": loguniform.rvs(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 24])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(training_hpo, data_dir=DATA_PATH, n_epoch=max_num_epochs),
        # resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        resources_per_trial={"cpu": 2},
        config=hp_config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # Storing the best hyperparamter configuration for future model training
    with open(os.path.join(DATA_PATH, 'best_trial_hp_search.pkl'), 'wb') as f:
        pickle.dump(best_trial, f)


RUN_LOCALLY = True
# Data sample size for Hyperparamter optimization
HP_TRAINING_DATA_SAMPLE_SIZE = 200
BATCH_SIZE = 4

if RUN_LOCALLY:
    # Your local machine paths
    MODEL_PATH = '/Users/kamrulhasan/projects/dl-models-public/models/cifar_net.pth'
    DATA_PATH = '/Users/kamrulhasan/projects/dl-models-public/data'
else:
    # Your server paths
    MODEL_PATH = '/mnt/home/hasanka/projects/dl-models-public/models/cifar_net.pth'
    DATA_PATH = '/mnt/home/hasanka/projects/dl-models-public/data'

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Load and normalize CIFAR10 data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                        download=True, transform=transform)
trainset.data = trainset.data[:HP_TRAINING_DATA_SAMPLE_SIZE]  # a small chunk of data

if __name__ == '__main__':
    # hyperpaterm optimization
    optimize_hyperparams()
    # model training
    # training()
    # # model evaluation on the test set
    # evaluate()
