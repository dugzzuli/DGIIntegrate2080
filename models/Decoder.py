# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear

from utils import *



class Decoder(nn.Module):
    def __init__(self,n_input=-1,inter_dims=[500,500,1000],active=False):
        super(Decoder,self).__init__()
        self.n_input=n_input
        self.inter_dims=inter_dims
        self.encoder = nn.Sequential()
        for i in range(len(self.inter_dims)):
            if i == 0:
                self.encoder.add_module('layer_{}'.format(i), nn.Linear(self.n_input, self.inter_dims[i]))
                if(active):
                    self.encoder.add_module('relu_{}'.format(i), nn.ReLU())
            else:
                if(i==-len(self.inter_dims)):
                    self.encoder.add_module('layer_{}'.format(i), nn.Linear(self.inter_dims[i - 1], self.inter_dims[i]))

                else:
                    self.encoder.add_module('layer_{}'.format(i), nn.Linear(self.inter_dims[i - 1], self.inter_dims[i]))
                    if (active):
                        self.encoder.add_module('relu_{}'.format(i), nn.ReLU())

    def forward(self, x):
        hidden=self.encoder(x)

        return hidden



