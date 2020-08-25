"""Deep Learning Methods.
"""
from .base import (train_val_test_split, train_model, train_epoch, test_epoch, test_model, generate_dataloader)


from .shallownet import ShallowConvNetv1, DeepConvNetv1
from .eegnet import EEGNetv1
from .hsnn import HSCNN
from .msnn import MSNN
