from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from early_stopping_pytorch.pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, predict_binding_site=False, binding_site_sizes=None):
        super(NeuralNet, self).__init__()
        self.predict_binding_site = predict_binding_site

        if predict_binding_site and binding_site_sizes is None:
            raise Exception('Provide the structure for binding_site hidden layers! Empty list for nothing')

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()

        layer_ordered_dict = OrderedDict()
        for i, hs in enumerate(hidden_sizes[1:]):
            layer_ordered_dict.update({f'fc{i + 2}': nn.Linear(hidden_sizes[i], hs),
                                       f'relu{i + 2}': nn.ReLU()})
        self.sequential = nn.Sequential(layer_ordered_dict)

        # if predict_binding_site:
        binding_site_layers_ordered_dict = OrderedDict()
        binding_site_layers_ordered_dict.update({'bs_fc1': nn.Linear(hidden_sizes[-1], binding_site_sizes[0])})

        for i, hs in enumerate(binding_site_sizes[1:]):
            binding_site_layers_ordered_dict.update(
                {f'bs_fc{i + 2}': nn.Linear(binding_site_sizes[i], hs),
                 f'bs_relu{i + 2}': nn.ReLU()})

        binding_site_layers_ordered_dict.update({'bs_out': nn.Linear(binding_site_sizes[-1], 3)})
        self.fc_out_binding = nn.Sequential(binding_site_layers_ordered_dict)
        # else:

        self.fc_out_non_binding = nn.Linear(hidden_sizes[-1], 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.sequential(out)
        if self.predict_binding_site:
            out = self.fc_out_binding(out)
        else:
            out = self.fc_out_non_binding(out)
        # out = self.fc_out(out)

        return self.softmax(out)


class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y.to_numpy()).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def train_val_test_nn(X_train_val, y_train_val, X_test, y_test, hidden_sizes, binding_site_hidden_sizes,
                      num_epochs=200, lr=0.01, patience=7):
    early_stopping = EarlyStopping(patience=patience, path='../data/models/checkpoint.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(len(X_train_val[0]), hidden_sizes, predict_binding_site=False,
                      binding_site_sizes=binding_site_hidden_sizes).to(device)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.85)
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    train_loader = DataLoader(MoleculeDataset(X_train, y_train))
    valid_loader = DataLoader(MoleculeDataset(X_val, y_val))
    test_loader = DataLoader(MoleculeDataset(X_test, y_test))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_losses = []
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model)

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
            torch.save(model.state_dict(), f'../data/models/model_{epoch + 1}.ckpt')

        if early_stopping.early_stop:
            print("Early stopping")
            break


    predicted = []
    with torch.no_grad():
        for features, label in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            predicted.append(torch.argmax(outputs.data, 1).data[0])

    return predicted


def transfer_train_val_test(X_train_val, y_train_val, X_test, y_test, hidden_sizes, binding_site_hidden_sizes,
                            model_checkpoint, num_epochs=200, lr=0.01, patience=7):
    early_stopping = EarlyStopping(patience=patience, path='../data/models/checkpoint_transfer.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(len(X_train_val[0]), hidden_sizes, predict_binding_site=True,
                      binding_site_sizes=binding_site_hidden_sizes).to(device)

    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.85)
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    train_loader = DataLoader(MoleculeDataset(X_train, y_train))
    valid_loader = DataLoader(MoleculeDataset(X_val, y_val))
    test_loader = DataLoader(MoleculeDataset(X_test, y_test))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_losses = []
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())
        valid_loss = np.average(valid_losses)
        early_stopping(valid_loss, model)

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
            torch.save(model.state_dict(), f'../data/models/model_{epoch + 1}.ckpt')

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('../data/models/checkpoint_transfer.pt'))
    model.eval()

    predicted = []
    with torch.no_grad():
        for features, label in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            predicted.append(torch.argmax(outputs.data, 1).data[0])

    return predicted
