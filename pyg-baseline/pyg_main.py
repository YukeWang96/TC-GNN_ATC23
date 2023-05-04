#!/usr/bin/env python3
import os.path as osp
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, AGNNConv
from torch_geometric.nn import GINConv
from torch.nn import Linear

from dataset import *



parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default="../tcgnn-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=200, help="number of epoches")
parser.add_argument("--run_GCN", action='store_true', help="run GCN otherwise GIN")

args = parser.parse_args()
print(args)

path = osp.join(args.dataDir, args.dataset+".npz")
# path = osp.join("../tcgnn-ae-graphs", args.dataset)
dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
data = dataset

run_GCN = args.run_GCN

if run_GCN:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True, normalize=False)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True, normalize=False)

        def forward(self):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

else:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.lin1 = torch.nn.Linear(dataset.num_features, 32)
            self.convs = torch.nn.ModuleList()
            for _ in range(4):
                self.convs.append(AGNNConv(requires_grad=False))
            self.lin2 = torch.nn.Linear(32, dataset.num_classes)

        def forward(self):
            x, edge_index = data.x, data.edge_index        
            x = F.relu(self.lin1(x))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

torch.cuda.synchronize()
start = time.perf_counter()
for epoch in tqdm(range(1, args.epochs + 1)):
    train()
torch.cuda.synchronize()
dur = time.perf_counter() - start

print("GCN (L2-H16) -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
print()