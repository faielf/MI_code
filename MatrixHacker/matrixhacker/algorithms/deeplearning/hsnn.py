import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .base import compute_flat_features, compute_out_size, MaxNormConstraint


class ScaleCNN(nn.Module):
    """Basic CNN for HSCNN.

    Return flattened single scale result.
    """
    def __init__(self, n_chan, n_sample, time_kernel=65):
        super(ScaleCNN, self).__init__()
        self.time_kernel = time_kernel
        self.conv_time = nn.Conv2d(1, 10, (1, self.time_kernel), stride=(1, 3), bias=False)
        w = compute_out_size(n_sample, self.time_kernel, stride=3)

        self.conv_space = nn.Conv2d(10, 10, (n_chan, 1), stride=1, bias=False)
        h = compute_out_size(n_chan, n_chan)

        self.max_pool = nn.MaxPool2d((1, 6), (1, 6))
        w = compute_out_size(w, 6, stride=6)

    def forward(self, X):
        # X is NCHW shape
        out = self.conv_time(X)

        out = self.conv_space(out)
        out = F.elu(out)

        out = self.max_pool(out)

        out = out.view(out.size()[0], -1)
        return out


class BandCNN(nn.Module):
    """Single frequency band CNN for HSCNN.
    
    Return flattened single freq band result (concate 3 scale results).
    """

    def __init__(self, n_chan, n_sample):
        super(BandCNN, self).__init__()

        self.scale_cnn1 = ScaleCNN(n_chan, n_sample, time_kernel=85)
        self.scale_cnn2 = ScaleCNN(n_chan, n_sample, time_kernel=65)
        self.scale_cnn3 = ScaleCNN(n_chan, n_sample, time_kernel=45)

    def forward(self, X):
        # X in NHW shape
        X = X.view(-1, 1, *X.size()[1:])

        out1 = self.scale_cnn1(X)
        out2 = self.scale_cnn2(X)
        out3 = self.scale_cnn3(X)

        out = torch.cat((out1, out2, out3), dim=-1)

        return out


class HSCNN(nn.Module):
    """A CNN with hybrid convolution scale.

    The paper [1]_ proposes a hybrid scale CNN network for EEG recognition.

    The newtwork is based on BCI Competition IV 2a/2b, which sampled at 250Hz. 
    The duration of a trial is 3.5s. 
    Only C3, Cz, C4 are recorded.

    Raw signal is filtered by three filters (4-7, 8-13, 13-32), then each frequency band is passed into CNN 
    with 3 time convolution kernel (85, 65, 45). Outputs are flattened and concated to a single feature vector.

    The paper uses elu as activation function, though they don't specify the location where it's applyed.
    In my implementation, elus are added following conv_space layer and fc_layer1.

    Some key parameters and tricks:  

    dropout probabilty                 0.8
    l2 regularizer(on fc_layer1)       0.01
    SGD optimizer                      0.1 decays every 10 epochs with exp decay rate of 0.9 (400 epochs total)
    
    The paper also uses data augmentation strategy, see the reference for more information.
    
    References
    ----------
    .. [1] Dai, Guanghai, et al. "HS-CNN: A CNN with hybrid convolution scale for EEG motor imagery classification." Journal of Neural Engineering (2019).
    """
    def __init__(self, n_chan, n_sample, n_class, n_band=1):
        super(HSCNN, self).__init__()
        self.band_cnns = nn.ModuleList([BandCNN(n_chan, n_sample) for i in range(n_band)])

        with torch.no_grad():
            fake_input = torch.zeros(1, 1, n_chan, n_sample)
            fake_output = self.band_cnns[0](fake_input)
            band_size = fake_output.size()[1]
        
        self.fc_layer1 = nn.Linear(band_size*n_band, 100)
        self.drop = nn.Dropout(0.8)
        self.fc_layer2 = nn.Linear(100, n_class)

    def forward(self, X):
        # X in (N, n_band, n_chan, n_sample)
        out = []
        for i, l in enumerate(self.band_cnns):
            out.append(l(X[:, i, ...]))

        out = torch.cat(out, dim=-1)
        
        out = self.fc_layer1(out)
        out = F.elu(out)
        out = self.fc_layer2(out)
        return out

