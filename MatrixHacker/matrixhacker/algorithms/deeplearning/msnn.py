# -*- coding: utf-8 -*-
"""multi-scale neural network.

Modified from https://github.com/DeepBCI/Deep-BCI
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .base import compute_flat_features, compute_out_size

class SeperableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode='zeros',
        depth_multiplier=1):
        super(SeperableConv2d, self).__init__()

        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels*depth_multiplier, kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode)
        
        self.pointwise_conv2d = nn.Conv2d(in_channels*depth_multiplier, out_channels, 1,
            groups=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
            padding_mode=padding_mode)

    def forward(self, X):
        out = self.depthwise_conv2d(X)
        out = self.pointwise_conv2d(out)
        return out


class BandSpaceLayer(nn.Module):

    def __init__(self, in_channels, out_band_channels, out_space_channels,
        kernel_band, kernel_space, padding=0, F_activation=None):
        super(BandSpaceLayer, self).__init__()

        self.F_activation = F_activation

        self.seperable_conv2d = SeperableConv2d(in_channels, out_band_channels, kernel_band,
            padding=padding)
        self.seperable_batch_norm = nn.BatchNorm2d(out_band_channels)

        self.space_conv2d = nn.Conv2d(out_band_channels, out_space_channels, kernel_space)
        self.space_batch_norm = nn.BatchNorm2d(out_space_channels)

    def forward(self, X):
        out1 = self.seperable_conv2d(X)
        if self.F_activation is not None:
            out1 = self.F_activation(out1)
        out1 = self.seperable_batch_norm(out1)

        out2 = self.space_conv2d(out1)
        if self.F_activation is not None:
            out2 = self.F_activation(out2)
        out2 = self.space_batch_norm(out2)

        return out1, out2


class MSNN(nn.Module):

    def __init__(self, n_chan, n_sample, n_class,
        fs=250):
        super(MSNN, self).__init__()
        self.time_conv2d = nn.Conv2d(1, 4, (1, fs//2))
        self.time_batch_norm = nn.BatchNorm2d(4)
        w = compute_out_size(n_sample, fs//2)

        self.bs_layer1 = BandSpaceLayer(4, 32, 32, (1, 15), (n_chan, 1),
            padding=7, F_activation=F.leaky_relu)
        w = compute_out_size(w, 15, padding=7)

        self.bs_layer2 = BandSpaceLayer(32, 32, 32, (1, 11), (n_chan, 1),
            padding=5, F_activation=F.leaky_relu)
        w = compute_out_size(w, 11, padding=5)

        self.bs_layer3 = BandSpaceLayer(32, 32, 32, (1, 5), (n_chan, 1),
            padding=2, F_activation=F.leaky_relu)
        w = compute_out_size(w, 5, padding=2)

        self.fc = nn.Linear(32*3, n_class)


    def forward(self, X):
        out = self.time_conv2d(X)
        out = F.leaky_relu(out)
        out = self.time_batch_norm(out)

        hidden1, out1 = self.bs_layer1(out)
        hidden2, out2 = self.bs_layer2(hidden1)
        hidden3, out3 = self.bs_layer3(hidden2)

        out = torch.cat((out1, out2, out3), dim=1)
        out = torch.mean(out, -1) # mean temporal features

        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out










        







        


        



