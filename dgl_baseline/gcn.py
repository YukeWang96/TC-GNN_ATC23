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
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(self.g, h)
        return h


class AGNN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):

        super(AGNN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.AGNN_layers = nn.ModuleList()
        self.activation = activation
        self.AGNN_layers.append(AGNNConv(in_dim, num_hidden, 8, activation=self.activation))
        
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.AGNN_layers.append(AGNNConv(num_hidden * heads[l-1], num_hidden, 8, activation=self.activation))

        # output projection
        self.AGNN_layers.append(AGNNConv(num_hidden * 8, num_classes, 1))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.AGNN_layers[l](self.g, h).flatten(1)

        # output projection
        logits = self.AGNN_layers[-1](self.g, h).mean(1)
        return logits