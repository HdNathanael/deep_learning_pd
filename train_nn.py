import numpy as np
import time
import fastprogress
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import KFold


def make_tensor(dat):
    '''
  Converts a 2d pandas dataframe to a tuple containing tensor
  Input: pandas dataframe
  Output: list containing tensors
  '''
    tups = []
    for jj in range(len(dat)):
        x = torch.tensor([dat.input.iloc[jj]]).float()
        y = torch.tensor([dat.label.iloc[jj]]).float()
        tups.append((x, y))
    return tups


def train_nn_reg(dataloader, model, optimiser, loss_fn, device):
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
    for x, y in dataloader:
        # initialise training mode
        optimiser.zero_grad()
        model.train()
        # forward pass
        if len(x.size()) > 2:
            x = x.view(-1, x.size()[-1] * x.size()[-2])
        y_hat = model(x.to(device))
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
    return np.mean(epoch_loss)


def validate_nn_reg(dataloader, model, loss_fn, device):
    '''
    Calculates validation error for validation dataset

    :param dataloader: dataloader object containing the validation data
    :param model: initialised Torch nn (nn.Module) to train
    :param loss_fn: Torch loss function
    :param device: device to use for training
    :return: validation loss for one epoch
    '''

    epoch_loss = []
    model.eval()  # initialise validation mode
    with torch.no_grad():  # disable gradient tracking
        for x, y in dataloader:
            # forward pass
            if len(x.size()) > 2:
                x = x.view(-1, x.size()[-1] * x.size()[-2])
            y_hat = model(x.to(device))
            # loss
            loss = loss_fn(y_hat, y.to(device))
            batch_loss = loss.detach().cpu().numpy()
            epoch_loss.append(batch_loss)
    return np.mean(epoch_loss)


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

    for epoch in fastprogress.progress_bar(range(n_epochs)):
        epoch_train_loss = train_nn_reg(train_loader, model, optimiser, loss_fn, device)
        train_loss.append(epoch_train_loss)

        if val_loader is not None:
            epoch_val_loss = validate_nn_reg(val_loader, model, loss_fn, device)
            val_loss.append(epoch_val_loss)

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds")
    if val_loader is not None:
        return train_loss, val_loss
    else:
        return train_loss


def k_fold_cv(n_folds, train_df, n_epochs, model, device, init_weights,
              optimiser=optim.Adam, learning_rate=1e-3,
              verbose=False, batch_size=4):
    '''
    Wrapper for k-fold Cross validation
    :param n_folds: number of folds
    :param idx: index of data
    :param train_df: Pandas dataframe object containing training data
    :param n_epochs: number of epochs
    :param model: initialised Torch nn (nn.Module) to train
    :param optimiser: Torch optimiser object
    :param loss_fn: Torch loss function
    :param device: device to use for training
    :param verbose:
    :param batch_size: batch size when creating dataloader objects
    :return: 2d array, 2d array, list, list
    '''
    print('Initialising Cross Validation')
    start_time = time.time()

    kfold = KFold(n_splits=n_folds, shuffle=True)
    train_loss = np.zeros((n_folds, n_epochs))
    val_loss = np.zeros((n_folds, n_epochs))

    train_tensor = make_tensor(train_df)
    # iterate over all folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_df)):
        print("")
        print(f"evaluating on fold {fold + 1} out of {n_folds}")

        # create training and validation indexes
        train_idx = torch.utils.data.SubsetRandomSampler(train_ids)
        val_idx = torch.utils.data.SubsetRandomSampler(val_ids)

        # pass indexes to dataloader
        train_loader = DataLoader(train_tensor, batch_size=batch_size, sampler=train_idx)
        val_loader = DataLoader(train_tensor, batch_size=batch_size, sampler=val_idx)

        # initialise model
        net = model
        net.apply(init_weights)
        net.to(device)

        # define loss function and optimiser
        loss_fn = nn.MSELoss()
        optim = optimiser(net.parameters(), lr=learning_rate)

        # run training and validation and store losses
        fold_train_loss, fold_val_loss = run_training(n_epochs, net, optim, loss_fn, device, train_loader, val_loader)

        # store losses
        train_loss[fold, :] = fold_train_loss
        val_loss[fold, :] = fold_val_loss

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 0).astype(int)
    print("")
    print(f"Completed Cross Validation after {time_elapsed} seconds")
    return train_loss, val_loss


def plot_cv_results(train_loss, val_loss):
    # calculate average training and validation loss
    avg_val_loss = np.mean(val_loss.T, axis=1)
    avg_train_loss = np.mean(train_loss.T, axis=1)

    #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    for jj in range(len(train_loss)):
        lab = "fold " + str(jj)
        ax1.semilogy(train_loss[jj], label=lab)
        ax1.set_ylabel("Log loss")
        ax1.set_xlabel("Epoch")
        ax1.set_title("Training loss")
        ax1.legend()

    for jj in range(len(val_loss)):
        lab = "fold " + str(jj)
        ax2.semilogy(val_loss[jj], label=lab)
        ax2.set_ylabel("Log loss")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Hold-out loss")
        ax2.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(18, 5))
    plt.semilogy(avg_val_loss, label="avg Hold-out loss")
    plt.semilogy(avg_train_loss, label="avg training loss")
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch")
    plt.title("Average Training and Hold-out loss")
    plt.legend()
    plt.show()


def plot_cv_hyper(train_loss, val_loss):
    # calculate average training and validation loss
    avg_val_loss = np.mean(val_loss, axis=1)
    avg_train_loss = np.mean(train_loss, axis=1)

    #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    for jj in range(len(avg_train_loss)):
        lab = "hyper comb " + str(jj)
        ax1.semilogy(avg_train_loss[jj], label=lab)
        ax1.set_ylabel("Log loss")
        ax1.set_xlabel("Epoch")
        ax1.set_title("Average Training loss")
        ax1.legend()

    for jj in range(len(avg_val_loss)):
        lab = "hyper comb " + str(jj)
        ax2.semilogy(avg_val_loss[jj], label=lab)
        ax2.set_ylabel("Log loss")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Average Validation loss")
        ax2.legend()

    plt.tight_layout()
    plt.show()


class MLP(nn.Module):

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
        return (x)


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


def init_weights(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.sparse_(m.weight,sparsity = 0.2)
        torch.nn.init.xavier_uniform_(m.weight)


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