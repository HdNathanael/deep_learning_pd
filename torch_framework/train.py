import numpy as np
import time

import fastprogress
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold

import torch_framework.aux as aux

# define functions to train the MLP
def train(dataloader, model, optimiser, loss_fn, device, classification = False):
    '''
    Trains one epoch of a neural network for clasification.

    :param dataloader: dataloader object containing the training data
    :param model: initialised Torch nn (nn.Module) to train
    :param optimiser: Torch optimiser object
    :param loss_fn: Torch loss function
    :param device: device to use for training
    :return: mean of epoch loss
    '''
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    for x, y in dataloader:
        # initialise training mode
        optimiser.zero_grad()
        model.train()
        # forward pass
        y_hat = model(x.to(device))
        if classification:
            # store number of correctly classified images
            epoch_correct += sum(y.to(device) == y_hat.argmax(dim=1)).cpu().numpy()
            epoch_total += len(y)
        # loss
        loss = loss_fn(y_hat, y.to(device))
        # Backpropagation
        loss.backward()
        # Update weights
        optimiser.step()
        # store batch loss
        batch_loss = loss.detach().cpu().numpy()  # move loss back to CPU
        epoch_loss.append(batch_loss)
    if classification:
        epoch_accuracy = float(epoch_correct / epoch_total)
        return np.mean(epoch_loss), epoch_accuracy
    else:
        return np.mean(epoch_loss)


def validate(dataloader, model, loss_fn, device, classification = False):
    '''
    Calculates validation error for validation dataset

    :param dataloader: dataloader object containing the validation data
    :param model: initialised Torch nn (nn.Module) to train
    :param loss_fn: Torch loss function
    :param device: device to use for training
    :return: validation loss for one epoch
    '''

    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    model.eval()  # initialise validation mode
    with torch.no_grad():  # disable gradient tracking
        for x, y in dataloader:
            # forward pass
            y_hat = model(x.to(device))
            if classification:
                # store number of correctly classified images
                epoch_correct += sum(y.to(device) == y_hat.argmax(dim=1))
                epoch_total += len(y)
            # loss
            loss = loss_fn(y_hat, y.to(device))
            batch_loss = loss.detach().cpu().numpy()
            epoch_loss.append(batch_loss)
    if classification:
        epoch_accuracy = epoch_correct / epoch_total
        return np.mean(epoch_loss), epoch_accuracy.cpu().numpy()
    else:
        return np.mean(epoch_loss)

# define wrappers for classification and regression
def run_training_cl(n_epochs, model, optimiser, loss_fn, device, train_loader,val_loader=None, early_stopper=None):
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
        epoch_train_loss, epoch_train_acc = train(train_loader, model, optimiser, loss_fn, device, classification=True)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        if val_loader is not None:
            epoch_val_loss, epoch_val_acc = validate(val_loader, model, loss_fn, device, classification=True)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)

        if early_stopper:
            early_stopper.update(epoch_val_loss, epoch_val_acc, model)
            if early_stopper.early_stop:
                early_stopper.load_checkpoint(model)
                print(f"Patience exhausted. Stopping early...")
                break

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds")
    if val_loader is not None:
        return train_loss, train_acc, val_loss, val_acc
    else:
        return train_loss, train_acc

def run_training_reg(n_epochs, model, optimiser, loss_fn, device, train_loader, val_loader=None):
    '''
    Wrapper for training and validation
    :param n_epchos: number of epochs to train
    :param model: initialised Torch nn (nn.Module) to train
    :param optimiser: Torch optimiser object
    :param train_loader: dataloader object for training data
    :param val_loader: dataloader object for validation data
    :param loss_fn: Torch loss function
    :param device: device to use (CPU, GPU)
    :return:
    '''
    print('Initialising training')
    start_time = time.time()
    train_loss = []
    val_loss = []

    model.to(device)

    for epoch in fastprogress.progress_bar(range(n_epochs)):
        epoch_train_loss = train(train_loader, model, optimiser, loss_fn, device)
        train_loss.append(epoch_train_loss)

        if val_loader is not None:
            epoch_val_loss = validate(val_loader, model, loss_fn, device)
            val_loss.append(epoch_val_loss)

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 0).astype(int)
    print(f"Finished training after {time_elapsed} seconds")
    if val_loader is not None:
        return train_loss, val_loss
    else:
        return train_loss

class EarlyStopper:
    '''Implements an early stopper, which stops trainings if the validation loss does not increase'''

    def __init__(self, path='checkpoint.pt', patience=10):
        '''

        :param path: path where the best model is saved, default: checkpoint.pt
        :param patience: number of epochs to wait before terminating training
        '''
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf
        self.best_acc = -np.Inf
        self.path = path

    @property
    def early_stop(self):
        return self.counter >= self.patience

    def update(self, val_loss, val_acc, model):
        if val_acc > self.best_acc:
            self.counter = 0
            self.save_checkpoint(model)
            self.best_loss = val_loss
            self.best_acc = val_acc
        else:
            self.counter += 1

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        print(f'Loading the best model...')
        print(f'Validation loss {self.best_loss:6f}')
        print(f'Validation accuracy {self.best_acc:6f}')
        model.load_state_dict(torch.load(self.path))
        model.eval()

def k_fold_cv(n_folds, train_df, n_epochs, model, device, init_weights,
              optimiser=optim.Adam, learning_rate=1e-3, wd = 0, batch_size=20, random_state = 1):
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

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state = random_state)
    train_loss = np.zeros((n_folds, n_epochs))
    val_loss = np.zeros((n_folds, n_epochs))

    train_tensor = aux.make_tensor(train_df)
    # iterate over all folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_df)):
        print("")
        print(f"evaluating on fold {fold + 1} out of {n_folds}")

        # create training and validation indexes
        train_idx = torch.utils.data.SubsetRandomSampler(train_ids)
        val_idx = torch.utils.data.SubsetRandomSampler(val_ids)

        # pass indexes to dataloader
        batch_size_train = len(train_idx)
        batch_size_val = len(val_idx)
        train_loader = DataLoader(train_tensor, batch_size=batch_size, sampler=train_idx)
        val_loader = DataLoader(train_tensor, batch_size=batch_size_val, sampler=val_idx)

        # initialise model
        net = model
        torch.manual_seed(random_state)
        net.apply(init_weights)
        net.to(device)

        # define loss function and optimiser
        loss_fn = nn.MSELoss()
        optim = optimiser(net.parameters(), lr=learning_rate, weight_decay=wd)

        # run training and validation and store losses
        fold_train_loss, fold_val_loss = run_training_reg(n_epochs, net, optim, loss_fn, device, train_loader, val_loader)

        # store losses
        train_loss[fold, :] = fold_train_loss
        val_loss[fold, :] = fold_val_loss

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 0).astype(int)
    print("")
    print(f"Completed Cross Validation after {time_elapsed} seconds")
    return train_loss, val_loss
#
# def tune_params(hyper_params, max_epochs=50):
#     n_mods = len(hyper_params[0])
#
#     train_loss_hyper = np.zeros((n_mods, max_epochs)) * np.nan
#     train_acc_hyper = np.zeros((n_mods, max_epochs)) * np.nan
#
#     val_loss_hyper = np.zeros((n_mods, max_epochs)) * np.nan
#     val_acc_hyper = np.zeros((n_mods, max_epochs)) * np.nan
#
#     for j in tqdm.tqdm(range(n_mods)):
#         act_fn = nn.ReLU
#         hidden_layer_params = dict(zip(list(range(hyper[0][j])), nodes[j]))
#         lr = hyper[1][j]
#         do_rate = hyper[2][j]
#         weight_decay = hyper[3][j]
#         patience = hyper[4][j]
#         Ni = 28 * 28
#         No = 10
#
#         mlp_hyper = MLP(Ni, No, hidden_layer_params, act_fn, dropout=do_rate)
#         mlp_hyper.apply(init_weights_kaiming)
#
#         loss_fn = nn.CrossEntropyLoss()
#         optimiser = optim.Adam(mlp_hyper.parameters(), lr=lr, weight_decay=weight_decay)
#
#         early_stopper = EarlyStopper(path='checkpoint.pt', patience=patience)
#         print("")
#         print(f'evaluate hyper parameter combination {j + 1}')
#         print(f'model structure: {hidden_layer_params}')
#         print(
#             f'learning_rate {lr:4f}, dropout_rate: {do_rate:4f}, weight_decay: {weight_decay:4f},patience = {patience}')
#
#         train_loss, train_acc, val_loss, val_acc = run_training(max_epochs, mlp_hyper, optimiser, loss_fn, device,
#                                                                 train_loader=train_loader, val_loader=val_loader,
#                                                                 early_stopper=early_stopper)
#
#         ep = len(train_loss)
#         train_loss_hyper[j, :ep] = train_loss
#         train_acc_hyper[j, :ep] = train_acc
#         val_loss_hyper[j, :ep] = val_loss
#         val_acc_hyper[j, :ep] = val_acc
#
#     return train_loss_hyper, train_acc_hyper, val_loss_hyper, val_acc_hyper
