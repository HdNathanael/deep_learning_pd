import numpy as np
import copy
import torch
import torch.nn as nn

def get_n_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def count_classes(dataset):
    class_count = np.zeros(10)
    for y in dataset.targets:
        lab = y.cpu().numpy()
        class_count[lab] += 1
    class_count = class_count/len(dataset)
    return class_count

def split_data(train_dataset, val_size=0.2):
    idx = list(range(len(train_dataset)))
    np.random.shuffle(idx)
    split = int(np.floor(val_size * len(train_dataset)))
    val_idx = idx[:split]

    mask = np.repeat(False, len(train_dataset))
    for i in val_idx:
        mask[i] = True

    tr_dataset, val_dataset = copy.deepcopy(train_dataset), copy.deepcopy(train_dataset)

    val_dataset.data = val_dataset.data[mask]
    val_dataset.targets = val_dataset.targets[mask]

    tr_dataset.data = tr_dataset.data[~mask]
    tr_dataset.targets = tr_dataset.targets[~mask]

    return tr_dataset, val_dataset

def make_tensor(dat):
    """
      Converts a 2d pandas dataframe to a tuple containing tensor
      Input: pandas dataframe
      Output: list containing tensors
    """
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

def init_weights_xavier_unif(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def init_weights_xavier_norm(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)