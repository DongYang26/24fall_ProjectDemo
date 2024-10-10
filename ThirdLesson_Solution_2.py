import copy
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn


class EqualUserSampler(object):
    def __init__(self, n, num_users) -> None:
        self.i = 0
        self.selected = n
        self.num_users = num_users
        self.get_order()

    def get_order(self):
        self.users = np.arange(self.num_users)

    def get_useridx(self):
        selection = list()
        for _ in range(self.selected):
            selection.append(self.users[self.i])
            self.i += 1
            if self.i >= self.num_users:
                self.get_order()
                self.i = 0
        return selection


os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
n_clients = 4
epochs = 10  # total epochs
local_epochs = 1  # local epochs of each user at an iteration
lr = 3e-3  # learning rate
cudaIdx = "cuda:0"  # GPU card index
device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
num_workers = 64  # workers for dataloader


# Load data (each client will load its own data in a real FL scenario)
def load_data(transform, datasets='MNIST'):
    if datasets == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=False, download=True, transform=transform)
    return train_dataset, test_dataset


# Split the dataset into 'n_clients' partitions
def partition_dataset(dataset, n_clients):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)


# CNN model definition
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # cifar10 use (3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # (16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 16 * 4 * 4)  # (-1, 16 * 5 * 5)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class FedAvgServer:  # used as a center
    def __init__(self, global_parameters):
        self.global_parameters = global_parameters

    def download(self, user_idx):
        local_parameters = []
        for i in range(len(user_idx)):
            local_parameters.append(copy.deepcopy(self.global_parameters))
        return local_parameters

    def upload(self, local_parameters):
        for i, (k, v) in enumerate(self.global_parameters.items()):
            tmp_v = torch.zeros_like(v)
            for j in range(len(local_parameters)):
                tmp_v += local_parameters[j][k]
            tmp_v = tmp_v / len(local_parameters)  # FedAvg
            self.global_parameters[k] = tmp_v


class Client:  # as a user
    def __init__(self, data_loader, user_idx):
        self.data_loader = data_loader
        self.user_idx = user_idx

    def train(self, model, learningRate, idx, global_model):  # training locally
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
        for epoch in range(epochs):
            for data, labels in self.data_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                # print(f"Client: {idx}({self.user_idx:2d})---- loss {loss.item():.4f}")


def activateClient(train_dataloaders, user_idx, server):
    local_parameters = server.download(user_idx)
    clients = []
    for i in range(len(user_idx)):
        clients.append(Client(train_dataloaders[user_idx[i]], user_idx[i]))
    return clients, local_parameters


def train(train_dataloaders, user_idx, server, global_model, learningRate):
    clients, local_parameters = activateClient(
        train_dataloaders, user_idx, server)
    for i in range(len(user_idx)):
        model = ConvNet().to(device)
        model.load_state_dict(local_parameters[i])
        model.train()
        clients[i].train(model, learningRate, i, global_model)
        local_parameters[i] = model.to(device).state_dict()
    server.upload(local_parameters)
    global_model.load_state_dict(server.global_parameters)


def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_main(n_clients=4):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # for cifar10
    train_dataset, test_dataset = load_data(transform)

    # Partition the dataset for each client
    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [DataLoader(dataset, batch_size=50, shuffle=True, num_workers=num_workers)
                      for dataset in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize global model for server
    global_model = ConvNet().to(device)
    global_parameters = global_model.state_dict()
    server = FedAvgServer(global_parameters)

    sampler = EqualUserSampler(n_clients, n_clients)

    for epoch in range(1, epochs + 1):  # start training
        print(f'Global Epoch {epoch}/{epochs}')
        user_idx = sampler.get_useridx()

        train(client_loaders, user_idx, server, global_model, lr)
        # Evaluate global model on test dataset
        test_accuracy = test(global_model, test_loader, device)
        print(
            f'Global Model Test Accuracy after round {epoch}: {test_accuracy:.4f}')
    # Save the final global model
    torch.save(global_model.state_dict(), 'federated_model.pth')
    print("Federated learning process completed.")


if __name__ == '__main__':

    train_main(n_clients)
