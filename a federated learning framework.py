'''
This is a federated learning framework for MNIST classification, on which you can build your own project model.

The framework provides:
Pre-implemented data loading functions, evaluation metrics, and the aggregation and distribution of model parameters in federated learning. Based on this framework, you can focus on designing your core functionality/model without worrying about the engineering aspects.
'''

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import copy


# design your CNN class, This class inherits from troch.nn.Module,so next init and forward functions need to be implemented.
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        

    def forward(self, x):
        
        
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



def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)



def client_update(client_model, optimizer, train_loader, device, epochs=1):
    client_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = client_model(data)  # design by yourself, decide by you CNN class designed.
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / len(train_loader)


def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[
                                     k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)


def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)  # design by yourself, decide by you CNN class designed.
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def federated_learning(n_clients, global_epochs, local_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  
    train_dataset, test_dataset = load_data(transform)


    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [DataLoader(dataset, batch_size=64, shuffle=True)
                      for dataset in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.0005)
                  for model in client_models]

    for global_epoch in range(global_epochs):
        print(f'Global Epoch {global_epoch + 1}/{global_epochs}')

        
        for client_idx in range(n_clients):
            client_update(client_models[client_idx], optimizers[client_idx],
                          client_loaders[client_idx], device, local_epochs)

        server_aggregate(global_model, client_models)
        
        test_accuracy = test_model(global_model, test_loader, device)
      
        print(f'Global Model Test Accuracy after round {global_epoch + 1}: {test_accuracy:.4f}')

    torch.save(global_model.state_dict(), 'federated_model.pth')
    print("Federated learning process completed.")


if __name__ == '__main__':
    federated_learning(n_clients=10, global_epochs=100, local_epochs=3)
