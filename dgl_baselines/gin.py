"""
GIN using DGL nn package
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from dgl.nn.pytorch import GINConv

class GIN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_hidden_layers,
                 activation,
                 dropout):
        super(GIN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        
        self.input_MLP =  nn.Linear(in_feats, n_hidden)
        self.hidden_MLP = nn.Linear(n_hidden, n_hidden)
        self.output_MLP = nn.Linear(n_hidden, n_classes)

        # input layer
        self.layers.append(GINConv(apply_func=self.input_MLP, aggregator_type='sum'))
        # hidden layers
        for i in range(n_hidden_layers - 1):
            self.layers.append(GINConv(apply_func=self.hidden_MLP, aggregator_type='sum'))
        # output layer
        self.layers.append(GINConv(apply_func=self.output_MLP, aggregator_type='sum'))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h