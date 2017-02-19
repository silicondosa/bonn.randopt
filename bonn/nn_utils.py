#!/usr/bin/env python

import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable as V
from torch.optim import  RMSprop
from torch.utils.data import TensorDataset, DataLoader


class TData(TensorDataset):

    def __init__(self, X, y):
        X = th.FloatTensor(X)
        y = th.FloatTensor(y)
        super(TData, self).__init__(X, y)

class FCNetwork(nn.Module):

    def __init__(self, num_in, layers=(5, 5)):
        super(FCNetwork, self).__init__()
        params = [nn.Linear(num_in, layers[0])]
        for i, l in enumerate(layers[1:]):
            layer = nn.Linear(layers[i-1], l)
            params.append(layer)
            setattr(self, 'l' + str(i), layer)
        layer = nn.Linear(layers[-1], 1)
        params.append(layer)
        setattr(self, 'last', layer)
        self.params = params

    def forward(self, x, drop=True):
        for l in self.params[:-2]:
            x = F.tanh(l(x))
        x = F.dropout(x, training=drop)
        x = F.tanh(self.params[-2](x))
        x = F.dropout(x, training=drop)
        x = self.params[-1](x)

        return x

def get_opt(params, lr):
    return RMSprop(params, lr=lr)

def get_loss():
    return nn.MSELoss()
