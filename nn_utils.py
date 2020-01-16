# -*- coding: utf-8 -*-
"""
Module containing standard utilities for neural nets:
    - cross validation routine
        cross_val(net, opt, epochs, mb_size, X_train, y_train, k_folds, device)
    - training routine
        train(net, epochs, optim, train_loader, device,
              loss_fn=F.binary_cross_entropy)
    - test routine
        eval(net, test_loader, device)
"""
import numpy as np
import torch
import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from copy import deepcopy


def cross_val(net, opt, epochs, mb_size, X_train, y_train, k_folds, device):
    """
    Cross validation implementation using ´k_folds´ number of folds.

    Parameters
    ----------
    net : Net
        Instance of the Net class, child of the torch.nn.Module.
    opt : torch.optim child
        Optimizer object.
    epochs : int
        Number of training epochs.
    mb_size : int
        Mini-batch size.
    X_train : pandas DataFrame
        Training set features.
    y_train : pandas Series
        Training set target vector.
    k_folds : int
        Number of cross validation folds.
    device : torch.device
        Torch device (cuda or cpu).

    Returns
    -------
    test_accs : numpy.ndarray
        Array containing accuracies of each fold.

    """
    init_state = deepcopy(net.state_dict())
    init_state_opt = deepcopy(opt.state_dict())
    fold_cntr = 0
    test_accs = np.zeros(k_folds)
    kf = KFold(n_splits=k_folds)
    for train_indices, test_indices in kf.split(X_train):
        net.load_state_dict(init_state)
        opt.load_state_dict(init_state_opt)
        train = TensorDataset(
            torch.Tensor(np.array(X_train.loc[train_indices])),
            torch.Tensor(np.array(y_train.loc[train_indices]))
        )
        val = TensorDataset(
            torch.Tensor(np.array(X_train.loc[test_indices])),
            torch.Tensor(np.array(y_train.loc[test_indices]))
        )
        train_loader = DataLoader(train, batch_size=mb_size, shuffle=False)
        val_loader = DataLoader(val, batch_size=mb_size, shuffle=False)
        net.train()
        str_desc = 'Fold ' + str(fold_cntr+1) + " / " + str(k_folds)
        for epoch in tqdm(range(1, epochs+1), desc=str_desc, unit=' ep'):
            running_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                mb = data.size(0)
                y_pred = net(data).reshape(-1)

                loss = F.binary_cross_entropy(y_pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()
        tqdm.write('')
        net.eval()
        running_loss = 0.0
        right_predictions = 0
        wrong_predictions = 0
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            mb = data.size(0)
            y_pred = net(data).reshape(-1)

            loss = F.binary_cross_entropy(y_pred, target)
            running_loss += loss.item()

            target_bool = target == 1
            proj = y_pred.cpu() > 0.5
            for j in range(mb):
                if proj[j] == target_bool[j].cpu():
                    right_predictions += 1
                else:
                    wrong_predictions += 1
        tqdm.write('Val Epoch, loss: {}, acc: {}, fold: {}'.format(
            running_loss/(i+1),
            100 * right_predictions / (right_predictions + wrong_predictions),
            fold_cntr+1)
        )
        test_accs[fold_cntr] = right_predictions / (right_predictions +
                                                    wrong_predictions)
        fold_cntr += 1
    return test_accs


def train(net, epochs, optim, train_loader, device,
          loss_fn=F.binary_cross_entropy):
    """
    Standard training routine.

    Parameters
    ----------
    net : Net
        Instance of the Net class, child of the torch.nn.Module.
    epochs : int
        Number of training epochs.
    optim : torch.optim child
        Optimizer object.
    train_loader : DataLoader
        Torch DataLoader for the training set.
    device : torch.device
        Torch device (cuda or cpu).
    loss_fn : Torch.nn.functional
        Torch function for the loss.

    Returns
    -------
    None.

    """
    net.train()
    for epoch in tqdm(range(1, epochs+1), desc='Training', unit=' ep'):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            # mb = data.size(0)
            y_pred = net(data).reshape(-1)

            loss = loss_fn(y_pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()


def eval(net, test_loader, device):
    """
    Standard test routine.

    Parameters
    ----------
    net : Net
        Instance of the Net class, child of the torch.nn.Module.
    test_loader : DataLoader
        Torch DataLoader for the test set.
    device : torch.device
        Torch device (cuda or cpu).

    Returns
    -------
    target_array : numpy.ndarray
        Prediction vector.

    """
    target_array = np.array([])
    net.eval()
    for i, [data] in enumerate(test_loader):
        data = data.to(device)
        y_pred = net(data)
        target = y_pred.detach().cpu().numpy()
        target_array = np.append(target_array, target)
    return target_array
