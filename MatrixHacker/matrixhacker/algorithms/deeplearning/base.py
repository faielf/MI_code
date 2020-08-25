import os, copy, time

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def compute_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def compute_out_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    return int((input_size + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)


def train_val_test_split(num_samples, test_size=0.1, val_size=0.1, stratify=None):
    train_size = 1 - test_size - val_size
    ix_train, ix_tmp = train_test_split(np.arange(num_samples),
        train_size=train_size, stratify=stratify)
    
    if stratify is not None:
        stratify = stratify[ix_tmp]

    ix_val, ix_test = train_test_split(ix_tmp, 
        train_size=val_size/(1-train_size), stratify=stratify)
    
    return ix_train, ix_val, ix_test


class MaxNormConstraint:
    """
    https://github.com/keras-team/keras/blob/master/keras/constraints.py#L22
    """
    def __init__(self, max_value=1, axis=None):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            norms = torch.sqrt(torch.sum(torch.pow(w, 2), dim=self.axis, keepdim=True))
            desired = torch.clamp(norms, 0, self.max_value)
            w *= (desired / (np.finfo(np.float).eps + norms))
            module.weight.data = w
        return module


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_epoch(dataloader, model, criterion, optimizer, device,
        regularizers=None):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        dataset_size += inputs.size(0)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        # update regularizer losses
        if regularizers is not None:
            for regularizer in regularizers:
                loss += regularizer(model.named_parameters())
        
        # update model
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    running_loss /= dataset_size

    return model, running_loss


def test_epoch(dataloader, model, criterion, device,
        regularizers=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            dataset_size += inputs.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # add regularizer losses
            if regularizers is not None:
                for regularizer in regularizers:
                    loss += regularizer(model.named_parameters())

            running_loss += loss.item() * inputs.size(0)
        
    running_loss /= dataset_size

    return model, running_loss


def test_model(dataloader, model, device, verbose=True):
    model.eval()

    dataset_size = 0

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            dataset_size += inputs.size(0)

            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)

            y_true.append(labels.cpu().detach().numpy())
            y_pred.append(preds.cpu().detach().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    running_acc = np.sum(y_true==y_pred)
    running_acc /= dataset_size

    cm = confusion_matrix(y_true, y_pred, np.unique(y_true))
    cm = cm / cm.sum(axis=1, keepdims=True)

    if verbose:
        print("Accuray: {:.4f}".format(running_acc))
        print("Labels: {}".format(np.unique(y_true)))
        print("Confusion Matrix:")
        print(cm)
        
    return running_acc, cm


def train_model(dataloaders, model, criterion, optimizer, device,
        epochs=10, regularizers=None, schedular=None, writer=None, estop=False, verbose=True):
    phases = list(dataloaders.keys())

    if writer is None:
        writer = SummaryWriter()

    if estop:
        estop_detector = EarlyStopping(verbose=verbose)

    # add model graph
    tmp_x, _ = dataloaders['train'].dataset[0]
    input_tensor = torch.zeros([1, *tmp_x.size()], dtype=tmp_x.dtype, device=device)
    writer.add_graph(model, input_tensor)

    best_epoch = 0
    best_loss = np.Inf
    best_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    since = time.time()
    
    train_acc = np.NaN
    val_acc = np.NaN
    test_acc = np.NaN

    for epoch in range(epochs):
        model, train_loss = train_epoch(dataloaders['train'], model, criterion, optimizer, device, regularizers=regularizers)
        train_acc, _ = test_model(dataloaders['train'], model, device, verbose=False)

        # update train tensorboard
        writer.add_scalars('loss', {'train': train_loss}, global_step=epoch)
        writer.add_scalars('accuracy', {'train': train_acc}, global_step=epoch)
        
        if train_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = train_loss
            best_acc = train_acc
            best_model_state = copy.deepcopy(model.state_dict())

            ck = {
                'epoch': best_epoch,
                'acc': best_acc,
                'loss': best_loss,
                'model_state': best_model_state,
                'optimizer': optimizer.state_dict()
            }

            torch.save(ck, 'checkpoint_best.pth.tar')

        if 'val' in phases:
            model, val_loss = test_epoch(dataloaders['val'], model, criterion, device, regularizers=regularizers)
            val_acc, _ = test_model(dataloaders['val'], model, device, verbose=False)

            writer.add_scalars('loss', {'val': val_loss}, global_step=epoch)
            writer.add_scalars('accuracy', {'val': val_acc}, global_step=epoch)

        if 'test' in phases:
            model, test_loss = test_epoch(dataloaders['test'], model, criterion, device, regularizers=regularizers)
            test_acc, _ = test_model(dataloaders['test'], model, device, verbose=False)

            writer.add_scalars('loss', {'test': test_loss}, global_step=epoch)
            writer.add_scalars('accuracy', {'test': test_acc}, global_step=epoch)

        if verbose:
            message = "Epoch {:d}/{:d} => Loss {:4f} => Train Acc {:2f} => Val Acc {:2f} => Test Acc {:2f}".format(epoch+1, epochs, train_loss, train_acc, val_acc, test_acc)
            print(message)

        if (estop 
            and ('val' in phases)
            and estop_detector(val_loss)):
            print("Early Stopped at epoch {:d}".format(epoch))
            break

        # adjust learning rate
        if schedular is not None:
            schedular.step()

    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Train Loss/Acc {:4f}/{:2f} at epoch {:d}".format(best_loss, best_acc, best_epoch))

    cks = torch.load('checkpoint_best.pth.tar')
    model.load_state_dict(cks['model_state'])
    model = model.cpu()

    return model


def generate_dataloader(X, y, dtype_x, dtype_y, batch_size=64, shuffle=True):
    X = torch.as_tensor(X, dtype=dtype_x)
    y = torch.as_tensor(y, dtype=dtype_y)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    return dataloader


def update_model_from_state_dict(model, state_dict):
    """For model transfer.
    """
    local_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in local_state_dict}
    
    local_state_dict.update(state_dict)
    model.load_state_dict(local_state_dict)
    return model


