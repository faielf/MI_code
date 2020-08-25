import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .base import compute_flat_features, compute_out_size, MaxNormConstraint


class ShallowConvNetv1(nn.Module):
    """
    https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    Assuming the input is a 2-second EEG signal sampled at 128Hz.

    Add max norm constraint on all convolutional layers and classification layer.

    Origin paper use convolution classifier instead of fc layer, which is the same thing when conv kernel equals to (h, w).

    Remove softmax layer with cross entropy loss in pytorch

                     vlawhern    original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    """
    def __init__(self, n_chan, n_sample, n_class, 
            n_time_filter=40, len_time_filter=31, n_space_filter=30, pool_size=35, pool_stride_size=7):
        super(ShallowConvNetv1, self).__init__()
        self.conv_time = nn.Conv2d(1, n_time_filter, (1, len_time_filter), bias=False, padding=(0, (len_time_filter-1)//2))
        self.constrain_conv_time = MaxNormConstraint(max_value=1, axis=(1, 2, 3))
        w = compute_out_size(n_sample, len_time_filter, padding=(len_time_filter-1)//2)
        self.bn_time = nn.BatchNorm2d(n_time_filter)

        self.conv_space = nn.Conv2d(n_time_filter, n_space_filter, (n_chan, 1), bias=False)
        self.constrain_conv_space = MaxNormConstraint(max_value=1, axis=(1, 2, 3))
        h = compute_out_size(n_chan, n_chan)

        self.bn_space = nn.BatchNorm2d(n_space_filter)
        self.pool_time = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride_size))
        self.drop = nn.Dropout(0.5)
        w = compute_out_size(w, pool_size, stride=pool_stride_size)
        self.fc = nn.Linear(n_space_filter*h*w, n_class)
        self.constrain_fc = MaxNormConstraint(max_value=0.5, axis=1)
        # self.conv_fc = nn.Conv2d(40, n_class, (h, w))
        

    def forward(self, X):
        # change view of X
        X = X.view(-1, 1, *X.size()[1:])
        # X = self.in_norm(X)

        self.conv_time = self.constrain_conv_time(self.conv_time)
        out = self.conv_time(X)
        out = self.bn_time(out)

        self.conv_space = self.constrain_conv_space(self.conv_space)
        out = self.conv_space(out)
        out = self.bn_space(out)
        out = torch.pow(out, 2)
        out = self.pool_time(out)
        out = torch.log(torch.clamp(out, min=1e-6)) # safe log avoiding log(0)
        out = self.drop(out)

        out = out.view(out.size()[0], -1)
        self.fc = self.constrain_fc(self.fc)
        out = self.fc(out)
        # out = self.softmax(out)

        return out


class DeepConvNetBlockv1(nn.Module):
    """
    https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    Basic block of DeepConvNetv1.
    """
    def __init__(self, in_channels, out_channels, kernel, pool_kernel, droprate=0.5):
        super(DeepConvNetBlockv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel)
        self.constrain_conv = MaxNormConstraint(max_value=2, axis=(1, 2, 3))

        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.drop = nn.Dropout(droprate)

    def forward(self, X):
        self.conv = self.constrain_conv(self.conv)
        out = self.conv(X)
        out = self.bn(out)
        out = F.elu(out)
        out = self.pool(out)
        out = self.drop(out)

        return out


class DeepConvNetv1(nn.Module):
    """
    https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    
    Assuming the input is a 2-second EEG signal sampled at 128Hz.

    Add max norm constraint on all convolutional layers and classification layer.

    Remove softmax layer with cross entropy loss in pytorch

                     vlawhern   original paper
    pool_size        1, 2       1, 3
    strides          1, 2       1, 3
    conv filters     1, 5       1, 10  
    """
    def __init__(self, n_chan, n_sample, n_class):
        super(DeepConvNetv1, self).__init__()
        self.conv_time = nn.Conv2d(1, 25, (1, 5))
        self.constrain_conv_time = MaxNormConstraint(max_value=2, axis=(1, 2, 3))
        w = compute_out_size(n_sample, 5)

        self.conv_space = DeepConvNetBlockv1(25, 25, (n_chan, 1), (1, 2))
        h = compute_out_size(n_chan, n_chan)
        w = compute_out_size(w, 2, stride=2)

        self.block1 = DeepConvNetBlockv1(25, 50, (1, 5), (1, 2))
        w = compute_out_size(w, 5, stride=1)
        w = compute_out_size(w, 2, stride=2)
        
        self.block2 = DeepConvNetBlockv1(50, 100, (1, 5), (1, 2))
        w = compute_out_size(w, 5, stride=1)
        w = compute_out_size(w, 2, stride=2)

        self.block3 = DeepConvNetBlockv1(100, 200, (1, 5), (1, 2))
        w = compute_out_size(w, 5, stride=1)
        w = compute_out_size(w, 2, stride=2)

        self.fc = nn.Linear(200*h*w, n_class)
        self.constrain_fc = MaxNormConstraint(max_value=0.5, axis=1)

    def forward(self, X):
        # change view of X
        X = X.view(-1, 1, *X.size()[1:])

        self.conv_time = self.constrain_conv_time(self.conv_time)
        out = self.conv_time(X)

        out = self.conv_space(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = out.view(out.size()[0], -1)
        self.fc = self.constrain_fc(self.fc)
        out = self.fc(out)

        return out
