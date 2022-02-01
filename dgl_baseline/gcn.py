"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, AGNNConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True, norm='none', weight=False, bias=False))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True, norm='none', weight=False, bias=False))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True, norm='none', weight=False, bias=False))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h