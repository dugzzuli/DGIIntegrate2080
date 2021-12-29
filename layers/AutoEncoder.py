# -*- coding: utf-8 -*-

# Distributed under terms of the MIT license.
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from utils import *



class Encoder(nn.Module):
    def __init__(self,n_input=784,inter_dims=[500,500,1000],n_z=10):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            nn.Linear(n_input,inter_dims[0]),
            nn.ReLU(),
            nn.Linear(inter_dims[0],inter_dims[1]),
            nn.ReLU(),
            nn.Linear(inter_dims[1],inter_dims[2]),
            nn.ReLU(),
        )

        self.mu_l=nn.Linear(inter_dims[2],n_z)
        # self.log_sigma2_l=nn.Linear(inter_dims[2],n_z)

    def forward(self, x):
        e=self.encoder(x)

        mu=self.mu_l(e)
        # log_sigma2=self.log_sigma2_l(e)
        self.mu = mu
        # self.log_sigma2 = log_sigma2

        return mu


class Decoder(nn.Module):
    def __init__(self,n_input=784,inter_dims=[500,500,1000],n_z=10):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            nn.Linear(n_z, inter_dims[-1]),
            nn.ReLU(),
            nn.Linear(inter_dims[-1], inter_dims[-2]),
            nn.ReLU(),
            nn.Linear(inter_dims[-2], inter_dims[-3]),
            nn.ReLU(),
            nn.Linear(inter_dims[-3], n_input),
            nn.Sigmoid()
        )
    def forward(self, z):
        x_pro=self.decoder(z)
        return x_pro

class AutoEncoder(nn.Module):
    def __init__(self,n_input=784,inter_dims=[500,500,1000],n_z=10):
        super(AutoEncoder,self).__init__()

        self.encoder = Encoder(n_input,inter_dims,n_z)
        self.decoder=Decoder(n_input,inter_dims,n_z)

    def forward(self, x):
        z = self.encoder(x)
        self.z = z
        x_pro=self.decoder(z)
        return x_pro,z
