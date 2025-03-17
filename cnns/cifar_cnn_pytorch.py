import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchsummary import summary
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


def accuracy(model, data_loader):
    """Accuracy estimation."""

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            # cpu vs gpu
            if device != 'cpu':
                images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct // total


def evaluate(estimate_training_accuracy=False):
    """model evaluation on the test set."""

    # Load previously trained model
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    if estimate_training_accuracy:
        # Evaluate on training data
        train_accuracy = accuracy(net, trainloader)
        print(f'Accuracy of the network on training images: {train_accuracy} %')

    # Evaluate on test data
    test_accuracy = accuracy(net, testloader)
    print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %')




def training(n_epoch=2, learning_rate=0.001, momentum=0.9):
    """Training a CNN model."""

    # CNN model instantiation
    net = Net()
    net.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Model/network training
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # cpu vs gpu
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
    # Save the model for future reference
    torch.save(net.state_dict(), MODEL_PATH)
    print('Finished Training')




RUN_LOCALLY = True
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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

if __name__ == '__main__':
    # model training
    training()
    # model evaluation on the test set
    evaluate(estimate_training_accuracy=True)

