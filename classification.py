import matplotlib.pyplot as plt
import numpy as np
import time
import fastprogress
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def train(dataloader, model, optimiser, loss_fn, device):
    '''
    Trains one epoch of a neural network for regression.

    :param dataloader: dataloader object containing the training data
    :param model: initialised Torch nn (nn.Module) to train
    :param optimiser: Torch optimiser object
    :param loss_fn: Torch loss function
    :param device: device to use for training
    :return: mean of epoch loss
    '''
    epoch_loss = []
    epoch_correct = 0
    epoch_total = 0
    for x, y in dataloader:
        # initialise training mode
        optimiser.zero_grad()
        model.train()
        # forward pass
        if len(x.size()) > 2:
            x = x.view(-1, x.size()[-1] * x.size()[-2])
        y_hat = model(x.to(device))
        # store number of correctly classified images
        epoch_correct += sum(y.to(device) == y_hat.argmax(dim = 1))
        epoch_total += len(y)

        # loss
        loss = loss_fn(y_hat, y.to(device))

        # Backpropagation
        # model.zero_grad()  # reset gradient
        loss.backward()
        # Update weights
        optimiser.step()

        # store batch loss
        batch_loss = loss.detach().cpu().numpy()  # move loss back to CPU
        epoch_loss.append(batch_loss)
    epoch_accuracy = float(epoch_correct/epoch_total)
    return np.mean(epoch_loss), epoch_accuracy

def validate(dataloader, model, loss_fn, device):
    '''
    Calculates validation error for validation dataset

    :param dataloader: dataloader object containing the validation data
    :param model: initialised Torch nn (nn.Module) to train
    :param loss_fn: Torch loss function
    :param device: device to use for training
    :return: validation loss for one epoch
    '''

    epoch_loss = []
    epoch_correct = 0
    epoch_total = 0
    model.eval()  # initialise validation mode
    with torch.no_grad():  # disable gradient tracking
        for x, y in dataloader:
            # forward pass
            if len(x.size()) > 2:
                x = x.view(-1, x.size()[-1] * x.size()[-2])
            y_hat = model(x.to(device))
            # store number of correctly classified images
            epoch_correct += sum(y.to(device) == y_hat.argmax(dim=1))
            epoch_total += len(y)
            # loss
            loss = loss_fn(y_hat, y.to(device))
            batch_loss = loss.detach().cpu().numpy()
            epoch_loss.append(batch_loss)
    epoch_accuracy = epoch_correct/epoch_total
    return np.mean(epoch_loss), epoch_accuracy

def run_training(n_epochs, model, optimiser, loss_fn, device, train_loader, val_loader=None, verbose=False):
    '''
    Wrapper for training and validation
    :param n_epchos: number of epochs to train
    :param model: initialised Torch nn (nn.Module) to train
    :param optimiser: Torch optimiser object
    :param train_loader: dataloader object for training data
    :param val_loader: dataloader object for validation data
    :param loss_fn: Torch loss function
    :param device: device to use (CPU, GPU)
    :param verbose:
    :return:
    '''
    print('Initialising training')
    start_time = time.time()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    model.to(device)
    for epoch in fastprogress.progress_bar(range(n_epochs)):
        epoch_train_loss, epoch_train_acc = train(train_loader, model, optimiser, loss_fn, device)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        if val_loader is not None:
            epoch_val_loss, epoch_val_acc = validate(val_loader, model, loss_fn, device)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds")
    if val_loader is not None:
        return train_loss, train_acc, val_loss, val_acc
    else:
        return train_loss, train_acc

class MLP_cl(nn.Module):

    def __init__(self, Ni, No, hidden_layer_params=None, act_fn=None, dropout=0):
        '''

        :param Ni: Number of input units
        :param No: Number of output units
        :param hidden_layer_params: dictionary containing the number of hidden units per hidden layer
        :param act_fn: Activation function (default is ReLU).
        '''
        super().__init__()  # initialise parent class

        if hidden_layer_params is None:
            n_hidden_layers = 0
            Nout = No
        else:
            n_hidden_layers = len(hidden_layer_params) - 1
            Nout = hidden_layer_params[0]

        if act_fn is None:
            act_fn = nn.ReLU

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=Ni, out_features=Nout))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(act_fn())

        # add hidden layers
        hidden_layers = 0
        while hidden_layers < n_hidden_layers:
            Nin = hidden_layer_params[hidden_layers]
            Nout = hidden_layer_params[hidden_layers + 1]
            self.layers.append(nn.Linear(in_features=Nin, out_features=Nout))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(act_fn())
            hidden_layers += 1

        # add output layer
        self.layers.append(nn.Linear(in_features=Nout, out_features=No))

    def forward(self, x):
        '''
        forward pass in the network
        :param x: input data
        :return: predicted values given the input data
        '''
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x, dim=1)

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.sparse_(m.weight,sparsity = 0.2)
        torch.nn.init.kaiming_normal_(m.weight)

def normalise(df):
    df_norm = copy.deepcopy(df)
    for j in range(len(df.columns)):
        col = df.iloc[:, j]
        mean = np.mean(col)
        sd = np.std(col)
        for it, x in enumerate(col):
            df_norm.iloc[it, j] = (x - mean) / sd

    return df_norm

def plot_class_results(train_loss,train_acc,val_loss,val_acc):
    plt.figure(figsize=(18, 5))
    plt.subplot(121)
    plt.semilogy(val_loss, label="Validation loss")
    plt.semilogy(train_loss, label="Training loss")
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch")
    plt.title("Training and Validation loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(train_acc,label = "Training Accuracy")
    plt.plot(val_acc,label = "Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()

    plt.show()