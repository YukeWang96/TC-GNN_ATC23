import os.path as osp
import argparse
import os
import sys
import time
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *

import TCGNN
from dataset import *
from gnn_conv import *
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--num_layers", type=int, default=2, help="num layers")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=10, help="number of epoches")
parser.add_argument("--model", type=str, default='gcn', help='GNN model', choices=['gcn', 'gin', 'agnn'])
args = parser.parse_args()
print(args)

#########################################
## Load Graph from files.
#########################################
dataset = args.dataset
# path = osp.join("/home/yuke/.graphs/orig", dataset)
path = osp.join("/home/yuke/.graphs/osdi-ae-graphs", dataset + ".npz")
dataset = TCGNN_dataset(path, args.dim, args.classes, load_from_txt=False)

num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index =  dataset.column_index 
row_pointers = dataset.row_pointers

#########################################
## Compute TC-GNN related graph MetaData.
#########################################
num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
edgeToRow = torch.zeros(num_edges, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)

# preprocessing for generating meta-information
start = time.perf_counter()
TCGNN.preprocess(column_index, row_pointers, num_nodes,  \
                BLK_H,	BLK_W, blockPartition, edgeToColumn, edgeToRow)
build_neighbor_parts = time.perf_counter() - start
print("Prep. (ms):\t{:.3f}".format(build_neighbor_parts*1e3))

column_index = column_index.cuda()
row_pointers = row_pointers.cuda()
blockPartition = blockPartition.cuda()
edgeToColumn = edgeToColumn.cuda()
edgeToRow = edgeToRow.cuda()

#########################################
## Build GCN and AGNN Model
#########################################
if args.model == "gcn":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden)

            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(GCNConv(args.hidden, args.hidden))
            
            self.conv2 = GCNConv(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            for Gconv  in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

elif args.model == "gin":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            
            self.conv1 = GINConv(dataset.num_features, args.hidden)
            self.hidden_layers = nn.ModuleList()
            for i in range(args.num_layers -  2):
                self.hidden_layers.append(GINConv(args.hidden, args.hidden))
            self.conv2 = GINConv(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            for Gconv in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

elif args.model == "agnn":
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = AGNNConv(dataset.num_features, args.hidden)
            self.hidden_layers = nn.ModuleList()
            for _ in range(args.num_layers -  2):
                self.hidden_layers.append(AGNNConv(args.hidden, args.hidden))
            self.conv2 = AGNNConv(args.hidden, dataset.num_classes)
            self.relu = nn.ReLU()

        def forward(self):
            x = dataset.x
            x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            for Gconv in self.hidden_layers:
                x = Gconv(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
                x = self.relu(x)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, dataset = Net().to(device), dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training 
def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[:], dataset.y[:])
    loss.backward()
    optimizer.step()

if __name__ == "__main__":

    # dry run.
    for epoch in range(1, 3):
        train()

    torch.cuda.synchronize()
    start_train = time.perf_counter()
    
    for _ in tqdm(range(1, args.epochs + 1)):
        train()

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train

    print("Train (ms):\t{:6.3f}".format(train_time*1e3/args.epochs))