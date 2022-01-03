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

def get_best_period(val_loss, val_acc):
    best = (min(val_loss), max(val_acc))
    best_epochs = (np.argmin(val_loss), np.argmax(val_acc))
    print(f"The model achieved the lowest validation loss in epoch {best_epochs[0]}: {best[0]:6f}")
    print(f"The model achieved the highest validation accuracy in epoch {best_epochs[1]}: {best[1]:6f}")

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)