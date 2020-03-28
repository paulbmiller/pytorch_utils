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
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from copy import deepcopy


def cross_val_c(net, opt, epochs, mb_size, X_train, y_train, k_folds, loss_fn,
                device):
    """
    Cross validation implementation using ´k_folds´ number of folds for
    classification (uses argmax at test time).

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
    X_train : numpy array
        Training set features.
    y_train : numpy array
        Training set target vector.
    k_folds : int
        Number of cross validation folds.
    loss_fn : torch.nn.functional
        Loss function
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
            torch.Tensor(np.array(X_train[train_indices])),
            torch.Tensor(np.array(y_train[train_indices]))
        )
        val = TensorDataset(
            torch.Tensor(np.array(X_train[test_indices])),
            torch.Tensor(np.array(y_train[test_indices]))
        )
        train_loader = DataLoader(train, batch_size=mb_size, shuffle=False)
        val_loader = DataLoader(val, batch_size=mb_size, shuffle=False)
        net.train()
        str_desc = 'Fold ' + str(fold_cntr+1) + "/" + str(k_folds)
        training_losses = np.zeros(epochs)
        for epoch in tqdm(range(1, epochs+1), desc=str_desc, unit='ep'):
            running_loss = 0.0
            train_nb_samples = 0
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                mb = data.size(0)
                train_nb_samples += mb
                y_pred = net(data)
                loss = loss_fn(y_pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()
            training_losses[i-1] = running_loss / train_nb_samples
        training_loss = running_loss
        net.eval()
        running_loss = 0.0
        right_predictions = 0
        total_predictions = 0
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            mb = data.size(0)
            y_pred = net(data)

            loss = loss_fn(y_pred, target)
            running_loss += loss.item()

            labels_pred = torch.argmax(y_pred, axis=1)
            labels_target = torch.argmax(target, axis=1)
            labels_right = labels_pred == labels_target
            labels_right = labels_right.sum().item()
            right_predictions += labels_right
            total_predictions += mb
        """
        Output the training loss for the last epoch of training, along with
        the validation loss, the accuracy and the fold number
        """
        tdqm_out = 'Val Epoch, train_loss: {:.4f}, val_loss: {:.4f}, '.format(
            training_loss,
            running_loss / total_predictions
        )
        tdqm_out += "acc: {:.3f}, fold: {}".format(
            100 * right_predictions / total_predictions,
            fold_cntr+1
        )
        tqdm.write(tdqm_out)
        test_accs[fold_cntr] = right_predictions / total_predictions
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
    losses = []
    for epoch in tqdm(range(1, epochs+1), desc='Training', unit=' ep'):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            # mb = data.size(0)
            y_pred = net(data)
            loss = loss_fn(y_pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() / len(target)
        losses.append(running_loss)


def evaluate(net, test_loader, device):
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

    target_tensor = torch.Tensor().to(device)
    net.eval()
    for i, data in enumerate(test_loader):
        data = data.to(device)
        y_pred = net(data)
        target_tensor = torch.cat((target_tensor, y_pred.detach()), 0)
    return target_tensor
