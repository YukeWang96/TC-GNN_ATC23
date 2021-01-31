import os.path as osp
import argparse
import os
import sys
import time
import torch
import math
import numpy as np 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.autograd.profiler as profiler
from scipy.sparse import *
from torch_geometric.datasets import Reddit

from dataset import *
from gcn_conv import *
from config import *
import GAcc

GCN = True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=40, help="number of epoches")
args = parser.parse_args()

#########################################
## Load Graph from files.
#########################################
dataset = args.dataset
if dataset == "reddit":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]
elif dataset in ['cora', 'pubmed', 'citeseer']:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
else:
    path = osp.join("/home/yuke/.graphs/orig", dataset)
    data = GAcc_dataset(path, args.dim, args.classes)
    dataset = data

# print(path)
#########################################
## Build Graph CSR.
#########################################
num_nodes = len(data.x)
num_edges = len(data.edge_index[1])
val = [1] * num_edges
start = time.perf_counter()
scipy_coo = coo_matrix((val, data.edge_index), shape=(num_nodes,num_nodes))
scipy_csr = scipy_coo.tocsr()
build_csr = time.perf_counter() - start
print("Build CSR: {:.3f}s ".format(build_csr))

column_index = torch.IntTensor(scipy_csr.indices)
row_pointers = torch.IntTensor(scipy_csr.indptr)

# degrees = (row_pointers[1:] - row_pointers[:-1]).tolist()
# degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()

#########################################
## Compute TC-GNN related graph MetaData.
#########################################
num_row_windows = (num_nodes + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_edges, dtype=torch.int)
edgeToRow = torch.zeros(num_edges, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)

# preprocessing for generating meta-information
start = time.perf_counter()
GAcc.preprocess(column_index, row_pointers, num_nodes, num_row_windows,  \
                BLK_H,	BLK_W, blockPartition, edgeToColumn, edgeToRow)
build_neighbor_parts = time.perf_counter() - start
print("Preprocess : {:.3f}s ".format(build_neighbor_parts))

column_index = column_index.cuda()
row_pointers = row_pointers.cuda()
blockPartition = blockPartition.cuda()
edgeToColumn = edgeToColumn.cuda()
edgeToRow = edgeToRow.cuda()

#########################################
## Build GCN and GAT Model.
#########################################
if GCN:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            GraphConv(dataset.num_features, args.hidden)
            GraphConv(args.hidden, dataset.num_classes)

        def forward(self):
            x = data.x
            x = F.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)
else:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GINConv(dataset.num_features, args.hidden)
            self.conv2 = GINConv(args.hidden, args.hidden)
            self.conv3 = GINConv(args.hidden, args.hidden)
            self.conv4 = GINConv(args.hidden, args.hidden)
            self.conv5 = GINConv(args.hidden, dataset.num_classes)

        def forward(self):
            x = data.x
            x = F.relu(self.conv1(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.relu(self.conv2(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.relu(self.conv3(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = F.relu(self.conv4(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow))
            x = self.conv5(x, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
            return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)


# Training 
def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Inference 
@torch.no_grad()
def test(profile=False):
    model.eval()
    if profile:
        with profiler.profile(record_shapes=True, use_cuda=True) as prof:
            with profiler.record_function("model_inference"):
                logits, accs = model(), []
        print(prof.key_averages().table())
    else:
        logits, accs = model(), []
    
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


if __name__ == "__main__":

    best_val_acc = test_acc = 0
    train_time_avg = []
    test_time_avg = []
    
    for epoch in range(1, args.epochs + 1):
        start_train = time.perf_counter()
        train()
        train_time = time.perf_counter() - start_train
        train_time_avg.append(train_time)
        # if epoch == 10:
        #     train_acc, val_acc, tmp_test_acc = test(profile=True)

        start_test = time.perf_counter()
        train_acc, val_acc, tmp_test_acc = test()
        test_time = time.perf_counter() - start_test
        test_time_avg.append(test_time)

        print("Epoch: {:2} Train (ms): {:6.3f} Test (ms): {:6.3f}".format(epoch, train_time * 1e3, test_time * 1e3))
        
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc
        # log = 'Epoch: {:03d}, Train: {:.4f}, Train-Time: {:.3f} ms, Test-Time: {:.3f} ms, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, train_acc, sum(time_avg)/len(time_avg) * 1e3, sum(test_time_avg)/len(test_time_avg) * 1e3, best_val_acc, test_acc))

    print("Avg. Train (ms): {:6.3f} Test (ms): {:6.3f}"\
            .format(epoch, sum(train_time_avg)/len(train_time_avg) * 1e3, sum(test_time_avg)/len(test_time_avg) * 1e3))
