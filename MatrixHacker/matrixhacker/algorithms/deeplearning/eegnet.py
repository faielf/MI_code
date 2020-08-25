import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .base import compute_flat_features, compute_out_size, MaxNormConstraint


class EEGNetv1(nn.Module):
    """
    https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    Assuming the input is a 1-second EEG signal sampled at 128Hz.
    EEGNet Settings:
    Parameter     vlawhern MatrixHacker
    kernel_time   32/64       31
    n_filter      8           8
    D             2           2
    
    Add max norm constraint on all convolutional layers and classification layer.

    Remove softmax layer with cross entropy loss in pytorch.

    Use 31 instead of 32(or 64) in conv_time kernel.

    Use 15 instead of 16 in separable_conv kernel.
    
    In EEGNet_SSVEP,
        srate       256
        n_class     12
        time_kernel (1, 256)
        F1          96
        D           1
        F2          96
    and remove fc constraint.
    """
    def __init__(self, n_chan, n_sample, n_class, 
            n_filter=8, D=2, kernel_time=31):
        super(EEGNetv1, self).__init__()

        self.conv_time = nn.Conv2d(1, n_filter, (1, kernel_time), 
            stride=1, padding=(0, (kernel_time-1)//2), bias=False)
        self.constrain_conv_time = MaxNormConstraint(max_value=1, axis=(1, 2, 3))
        h = n_chan
        w = n_sample
        self.bn_time = nn.BatchNorm2d(n_filter)

        # D = 4 # depthwise multiplier
        self.depthwise_conv = nn.Conv2d(n_filter, n_filter*D, (n_chan, 1), 
            groups=n_filter, bias=False)
        self.constrain_depthwise_conv = MaxNormConstraint(max_value=1, axis=(1, 2, 3))
        self.bn_depthwise = nn.BatchNorm2d(n_filter*D)
        self.pool_depthwise = nn.AvgPool2d((1, 4), stride=(1, 4))
        self.drop_depthwise = nn.Dropout(0.5)
        h = compute_out_size(h, n_chan)
        w = compute_out_size(w, 4, stride=4)

        self.separable_conv = nn.Conv2d(n_filter*D, n_filter*D, (1, 15), 
            padding=(0, 7), bias=False)
        self.bn_separable = nn.BatchNorm2d(n_filter*D)
        self.pool_separable = nn.AvgPool2d((1, 8), stride=(1, 8))
        self.drop_separable = nn.Dropout(0.5)
        w = compute_out_size(w, 8, stride=8)

        self.fc = nn.Linear(n_filter*D*h*w, n_class)
        self.constrain_fc = MaxNormConstraint(max_value=0.25, axis=1)
        
    def forward(self, X):
        X = X.view(-1, 1, *X.size()[1:])

        self.conv_time = self.constrain_conv_time(
            self.conv_time)
        out = self.conv_time(X)
        out = self.bn_time(out)

        self.depthwise_conv = self.constrain_depthwise_conv(
            self.depthwise_conv)
        out = self.depthwise_conv(out)
        out = self.bn_depthwise(out)
        out = F.elu(out)
        out = self.pool_depthwise(out)
        out = self.drop_depthwise(out)

        out = self.separable_conv(out)
        out = self.bn_separable(out)
        out = F.elu(out)
        out = self.pool_separable(out)
        out = self.drop_separable(out)

        out = out.view(out.size()[0], -1)
        self.fc = self.constrain_fc(self.fc)
        out = self.fc(out)

        return out
