import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings('ignore')


def train_model(model: nn.Module, device: str, criterion: F,
                optimizer: torch.optim, scheduler: torch.optim,
                train_dl: DataLoader, val_dl: DataLoader,
                epochs_num: int, suptitle_name: str = None) -> tuple[int, float, nn.Module]:
    """
    Model training.

    :param model: source untrained model;
    :param device: cpu or gpu;
    :param criterion: loss function;
    :param optimizer: tool for loss minimization;
    :param scheduler: decrement lr;
    :param train_dl: data loader for train data;
    :param val_dl: data loader for validation data;
    :param epochs_num: epochs number;
    :param suptitle_name: general heading for subplots;
    :return: best epoch number, minimum loss at this epoch
    and therefore best model during training.
    """
    plt.style.use('dark_background')
    model.to(device)

    best_model = copy.deepcopy(model)
    best_loss = np.inf
    best_epoch = -1

    loss_history_train = {}
    loss_history_val = {}

    for epoch in range(1, epochs_num + 1):
        model.train()
        scheduler.step()
        cur_loss = 0

        for X_train, y_train in train_dl:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            model.zero_grad()

            pred = model(X_train)
            loss_val = criterion(pred, y_train)
            loss_val.backward()
            optimizer.step()

            cur_loss += loss_val.item()

        cur_mean_train_loss = cur_loss / len(train_dl)
        loss_history_train[epoch] = cur_mean_train_loss

        model.eval()
        cur_loss = 0

        with torch.no_grad():
            for X_val, y_val in val_dl:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                pred = model(X_val)
                loss_val = criterion(pred, y_val)

                cur_loss += loss_val.item()

        cur_mean_val_loss = cur_loss / len(val_dl)
        loss_history_val[epoch] = cur_mean_val_loss

        if cur_mean_val_loss < best_loss:
            best_loss = cur_mean_val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        display.clear_output(wait=True)
        plt.figure(figsize=(25, 8))

        plt.subplot(1, 2, 1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        sns.lineplot(x=loss_history_train.keys(), y=loss_history_train.values(), label='train_loss')
        plt.title('Train data')

        plt.subplot(1, 2, 2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        sns.lineplot(x=loss_history_val.keys(), y=loss_history_val.values(), label='val_loss')
        plt.title('Validation data')

        if suptitle_name:
            plt.suptitle(suptitle_name)

        plt.show();

    return best_epoch, best_loss, best_model


def predict_model(model: nn.Module, dl: DataLoader, device: str) -> np.ndarray:
    """
    Make a model prediction.

    :param model: current model;
    :param dl: current data loader;
    :param device: cpu or gpu;
    :return: matrix with predictions.
    """
    pred = []
    model.to(device)
    model.eval

    with torch.no_grad():
        for X, _ in tqdm(dl):
            X = X.to(device)
            cur_pred = model(X).detach().cpu().numpy()
            pred.append(cur_pred)

    res = np.concatenate(pred, 0).argmax(-1)
    return res
