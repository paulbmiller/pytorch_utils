# -*- coding: utf-8 -*-
"""
Module for a neural net, to be able to add layers dynamically and build
different neural nets.
"""
import torch.nn as nn


# Create the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.modules = []
        self.model = nn.Sequential()
        self.optim_class = None
        self.optim = None

    def forward(self, x):
        return self.model(x)

    def add_layer(self, module):
        self.modules.append(module)
        self.update()

    def update(self):
        self.model = nn.Sequential(*self.modules)
        if self.optim_class:
            self.init_optim(self.optim_class, self.lr)

    def add_linear(self, in_features, out_features, bias=True):
        self.add_layer(nn.Linear(in_features, out_features, bias))

    def add_dropout(self, p=0.5, inplace=True):
        self.add_layer(nn.Dropout2d(p, inplace))

    def add_relu(self, inplace=True):
        self.add_layer(nn.ReLU(inplace))

    def add_leakyrelu(self, negative_slope=0.01, inplace=True):
        self.add_layer(nn.LeakyReLU(negative_slope, inplace))

    def add_sigmoid(self):
        self.add_layer(nn.Sigmoid())        

    def add_tanh(self):
        self.add_layer(nn.Tanh())

    def add_conv_layer(self, in_channels, out_channels, kernel_size, stride=1,
                       padding=0, dilation=1, groups=1, bias=True,
                       padding_mode='zeros'):
        self.add_layer(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation, groups, bias, padding_mode)
            )

    def init_optim(self, optimizer, lr):
        if not self.optim_class:
            self.optim_class = optimizer
            self.lr=lr
        self.optim = optimizer(self.parameters(), lr=lr)

    def get_optim(self):
        if self.optim:
            return self.optim
        else:
            print("Optimizer not initialized (use init_optim(optim, lr))")
            return None
