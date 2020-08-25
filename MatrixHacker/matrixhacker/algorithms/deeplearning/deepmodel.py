# -*- coding: utf-8 -*-
"""Awesome deep learning models, implemented in pytorch.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .base import compute_flat_features, compute_out_size, MaxNormConstraint


class RCL2dv1(nn.Module):
    """
    https://github.com/Mrswolf/kaggle_EEG/blob/master/models/fake_shuffle_len3584_bs_c1r4p5_f9n192r35p2.py
    https://github.com/Mrswolf/gumpy-deeplearning/blob/master/models/rcnn.py
    https://github.com/Mrswolf/Deep-BCI/tree/master/1_Intelligent_BCI/Spatio_Temporal_RCNN_for_EEG
    I don't understand original rCNN code. This is a prototype layer based on gumpy code.
    Developing
    """
    def __init__(self, in_chan, out_chan, feed_kernel, hidden_kernel, T=3):
        super(RCL2dv1, self).__init__()
        self.T = T
        self.feed_layer = nn.Conv2d(in_chan, out_chan, feed_kernel, 
            stride=1, padding=((feed_kernel[0]-1)//2, (feed_kernel[1]-1)//2))
        self.bn_feed = nn.BatchNorm2d(out_chan)
        
        self.hidden_layers = nn.ModuleList(modules=[
            nn.Conv2d(out_chan, out_chan, hidden_kernel, 
                stride=1, padding=((hidden_kernel[0]-1)//2, (hidden_kernel[1]-1)//2)) for _ in range(self.T)
            ]
        )
        self.hidden_bn_layers = nn.ModuleList(modules=[
            nn.BatchNorm2d(out_chan, eps=1e-6) for _ in range(self.T)
            ]
        )

    def forward(self, X):
        out_feed = self.feed_layer(X)
        out = self.bn_feed(out_feed)
        out = F.elu(out)

        for i in range(self.T):
            out = self.hidden_layers[0](out) + out_feed
            out = self.hidden_bn_layers[0](out)
            out = F.elu(out)

        return out


class RecurrentCNNv1(nn.Module):
    """
    https://github.com/Mrswolf/kaggle_EEG/blob/master/models/fake_shuffle_len3584_bs_c1r4p5_f9n192r35p2.py
    https://github.com/Mrswolf/gumpy-deeplearning/blob/master/models/rcnn.py
    I don't understand original rCNN code. This is a prototype layer based on gumpy code.
    Developing
    """
    def __init__(self, n_chan, n_sample, n_class, n_filter):
        super(RecurrentCNNv1, self).__init__()
        w = n_sample
        h = 1
        self.conv1 = nn.Conv2d(n_chan, n_filter, (1, 31), padding=(0, 15), stride=1)
        self.bn1 = nn.BatchNorm2d(n_filter, eps=1e-6)
        self.pool1 = nn.MaxPool2d((1, 2), (1, 2))
        self.drop1 = nn.Dropout(0.5)
        w = compute_out_size(w, 2, stride=2)

        self.rcl1 = RCL2dv1(n_filter, n_filter, (1, 1), (1, 5), T=3)
        self.rcl_pool1 = nn.MaxPool2d((1, 2), (1, 2))
        self.rcl_drop1 = nn.Dropout(0.5)
        w = compute_out_size(w, 2, stride=2)

        self.rcl2 = RCL2dv1(n_filter, n_filter, (1, 1), (1, 5), T=3)
        self.rcl_pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.rcl_drop2 = nn.Dropout(0.5) 
        w = compute_out_size(w, 2, stride=2)

        self.rcl3 = RCL2dv1(n_filter, n_filter, (1, 1), (1, 5), T=3)
        self.rcl_pool3 = nn.MaxPool2d((1, 2), (1, 2))
        self.rcl_drop3 = nn.Dropout(0.5)
        w = compute_out_size(w, 2, stride=2)

        self.fc = nn.Linear(n_filter*h*w, n_class)
        
    def forward(self, X):
        X = X.view(X.shape[0], X.shape[1], 1, -1)

        out = self.conv1(X)
        out = self.bn1(out)
        out = F.elu(out)
        out = self.pool1(out)
        out = self.drop1(out)

        out = self.rcl1(out)
        out = self.rcl_pool1(out)
        out = self.rcl_drop1(out)

        out = self.rcl2(out)
        out = self.rcl_pool2(out)
        out = self.rcl_drop2(out)

        out = self.rcl3(out)
        out = self.rcl_pool3(out)
        out = self.rcl_drop3(out)

        out = out.view(out.size()[0], -1)
        out = self.fc(out)

        return out   


class VanillaRNN(nn.Module):
    """
    Developing
    """
    def __init__(self, n_chan, n_sample, n_class, 
        n_hidden_size=128, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.n_hidden_size = n_hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(n_chan, n_hidden_size, num_layers=num_layers, nonlinearity='tanh')
        self.fc = nn.Linear(self.n_hidden_size*n_sample, n_class)

    def forward(self, X):
        X = X.permute(2, 0, 1)
        out, hn = self.rnn(X)
        out = out.permute(1, 0, 2)
        out = out.contiguous()
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


class FilterRNN(nn.Module):
    """
    Developing
    """
    def __init__(self, n_chan, n_sample, n_class, 
        n_hidden_size=128, num_layers=1, n_time_filter=20):
        super(FilterRNN, self).__init__()
        self.n_hidden_size = n_hidden_size
        self.num_layers = num_layers

        self.conv_time = nn.Conv2d(1, n_time_filter, (1, 13), 
            stride=1, padding=(0, 6), bias=False)
        self.constrain_conv_time = MaxNormConstraint(max_value=1, axis=(1, 2, 3))
        self.bn_time = nn.InstanceNorm2d(n_time_filter)

        self.rnn = nn.RNN(n_chan*n_time_filter, n_hidden_size, num_layers=num_layers, nonlinearity='tanh', dropout=0.5)
        self.fc = nn.Linear(self.n_hidden_size*n_sample, n_class)

    def forward(self, X):
        X = X.view(-1, 1, *X.size()[1:])

        self.conv_time = self.constrain_conv_time(
            self.conv_time)
        out = self.conv_time(X)
        out = self.bn_time(out)        
        
        out = out.view(out.shape[0], -1, out.shape[-1])
    
        out = out.permute(2, 0, 1)
        out, hn = self.rnn(out)
        out = out.permute(1, 0, 2)
        out = out.contiguous()
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


