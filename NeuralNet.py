# -*- coding: utf-8 -*-
"""
Module for a neural net, to be able to add layers dynamically and build
different neural nets.
"""
import torch.nn as nn


# Create the model
class Net(nn.Module):
    layers = []

    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential()

    def forward(self, x):
        return self.model(x)

    def update(self):
        pass
