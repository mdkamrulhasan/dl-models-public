import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# from models import MyVanillaCNNNet as Net
from models import Net24x24 as Net

from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import pandas as pd


# 1. Define a custom Data Loader
class NumpyDataset(Dataset):
    def __init__(self, X, Y=None, transform=None):
        self.X = torch.from_numpy(X).float()  # convert to float tensor
        if Y is not None:
            self.Y = torch.from_numpy(Y).long()  # convert to long tensor
        else:
            self.Y = torch.from_numpy(np.zeros(len(self.X))).long()
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def create_directory(path):
    """
    Creates a directory if it does not already exist.
    Displays an appropriate message on success or error.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully (or already exists).")
    except PermissionError:
        print(f"Error: Permission denied while creating '{path}'.")
    except OSError as e:
        print(f"Error: Failed to create directory '{path}'. {e}")


def accuracy(model, data_loader):
    """Accuracy estimation."""

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs

    output_labels = []

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
            output_labels += predicted.tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct // total, output_labels


def evaluate(estimate_training_accuracy=False):
    """model evaluation on the test set."""

    # Load previously trained model
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(MODEL_PATH, '24x24_challange_net.pth'), weights_only=True))

    if estimate_training_accuracy:
        # Evaluate on training data
        train_accuracy, _ = accuracy(net, trainloader)
        print(f'Accuracy of the network on training images: {train_accuracy} %')

    # Evaluate on test data
    test_accuracy, test_output_labels = accuracy(net, testloader)
    # print(f'Accuracy of the network on the 10000 test images: {test_accuracy} %')

    return test_output_labels


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
    torch.save(net.state_dict(), os.path.join(MODEL_PATH, '24x24_challange_net.pth'))
    print('Finished Training')


RUN_LOCALLY = False
BATCH_SIZE = 20

if RUN_LOCALLY:
    # Your local machine paths
    MODEL_PATH = '/Users/hasanka/projects/dl-models-public/models'
    DATA_PATH = '/Users/hasanka/projects/dl-models-public/data/24x24-challenge'
else:
    # Your server paths
    MODEL_PATH = '/mnt/home/hasanka/projects/dl-models-public/models'
    DATA_PATH = '/mnt/home/hasanka/projects/dl-models-public/data/24x24-challenge'

create_directory(MODEL_PATH)
create_directory(DATA_PATH)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Training data
with open(os.path.join(DATA_PATH, 'train_X_y.pkl'), 'rb') as f:
    X_train, y_train = pickle.load(f)

# Test data
with open(os.path.join(DATA_PATH, 'test_X.pkl'), 'rb') as f:
    X_test = pickle.load(f)

# Reshaping data matrices
X_train = np.transpose(X_train.astype(int), (0, 3, 1, 2))
X_test = np.transpose(X_test.astype(int), (0, 3, 1, 2))

# Load and normalize  data
transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = NumpyDataset(X_train, y_train.flatten(), transform=transform)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = NumpyDataset(X_test, transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

if __name__ == '__main__':
    # model training
    training()
    # model evaluation on the test set
    test_output_labels = evaluate(estimate_training_accuracy=True)

    # Pack results in a pandas dataframe
    test_predictions = pd.DataFrame({
        'rowId': range(0, len(test_output_labels)),
        'label': test_output_labels})

    test_predictions.to_csv(os.path.join(DATA_PATH, 'my_cnn_results.csv'), index=False)
