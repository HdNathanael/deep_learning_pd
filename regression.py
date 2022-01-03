import numpy as np
import copy

import torch

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

def normalise(df):
    df_norm = copy.deepcopy(df)
    for j in range(len(df.columns)):
        col = df.iloc[:, j]
        mean = np.mean(col)
        sd = np.std(col)
        for it, x in enumerate(col):
            df_norm.iloc[it, j] = (x - mean) / sd

    return df_norm

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
