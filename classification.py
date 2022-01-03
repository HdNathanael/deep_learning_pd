import matplotlib.pyplot as plt
import numpy as np
import time
import fastprogress
import copy

import torch
import torch.nn as nn


def plot_class_results(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(18, 5))
    plt.subplot(121)
    plt.semilogy(val_loss, label="Validation loss")
    plt.semilogy(train_loss, label="Training loss")
    plt.ylabel("Log Loss")
    plt.xlabel("Epoch")
    plt.title("Training and Validation loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(train_acc, label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()

    plt.show()


def plot_class_hyper(train_loss_hyper, train_acc_hyper, val_loss_hyper, val_acc_hyper):
    plt.figure(figsize=(18, 10))
    plt.subplot(221)
    for j in range(len(train_loss_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(train_loss_hyper[j], label=lab)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(222)
    for j in range(len(val_loss_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(val_loss_hyper[j], label=lab)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Validation Loss")
    plt.legend()

    plt.subplot(223)
    for j in range(len(train_acc_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(train_acc_hyper[j], label=lab)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training Accuracy")
    plt.legend()

    plt.subplot(224)
    for j in range(len(val_acc_hyper)):
        lab = "hyper comb" + str(j)
        plt.plot(val_acc_hyper[j], label=lab)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def tune_params(hyper_params, max_epochs=50):
    n_mods = len(hyper_params[0])

    train_loss_hyper = np.zeros((n_mods, max_epochs)) * np.nan
    train_acc_hyper = np.zeros((n_mods, max_epochs)) * np.nan

    val_loss_hyper = np.zeros((n_mods, max_epochs)) * np.nan
    val_acc_hyper = np.zeros((n_mods, max_epochs)) * np.nan

    for j in tqdm.tqdm(range(n_mods)):
        act_fn = nn.ReLU
        hidden_layer_params = dict(zip(list(range(hyper[0][j])), nodes[j]))
        lr = hyper[1][j]
        do_rate = hyper[2][j]
        weight_decay = hyper[3][j]
        patience = hyper[4][j]
        Ni = 28 * 28
        No = 10

        mlp_hyper = MLP(Ni, No, hidden_layer_params, act_fn, dropout=do_rate)
        mlp_hyper.apply(init_weights_kaiming)

        loss_fn = nn.CrossEntropyLoss()
        optimiser = optim.Adam(mlp_hyper.parameters(), lr=lr, weight_decay=weight_decay)

        early_stopper = EarlyStopper(path='checkpoint.pt', patience=patience)
        print("")
        print(f'evaluate hyper parameter combination {j + 1}')
        print(f'model structure: {hidden_layer_params}')
        print(
            f'learning_rate {lr:4f}, dropout_rate: {do_rate:4f}, weight_decay: {weight_decay:4f},patience = {patience}')

        train_loss, train_acc, val_loss, val_acc = run_training(max_epochs, mlp_hyper, optimiser, loss_fn, device,
                                                                train_loader=train_loader, val_loader=val_loader,
                                                                early_stopper=early_stopper)

        ep = len(train_loss)
        train_loss_hyper[j, :ep] = train_loss
        train_acc_hyper[j, :ep] = train_acc
        val_loss_hyper[j, :ep] = val_loss
        val_acc_hyper[j, :ep] = val_acc

    return train_loss_hyper, train_acc_hyper, val_loss_hyper, val_acc_hyper


def get_best_period(val_loss, val_acc):
    best = (min(val_loss), max(val_acc))
    best_epochs = (np.argmin(val_loss), np.argmax(val_acc))
    print(f"The model achieved the lowest validation loss in epoch {best_epochs[0]}: {best[0]:6f}")
    print(f"The model achieved the highest validation accuracy in epoch {best_epochs[1]}: {best[1]:6f}")

