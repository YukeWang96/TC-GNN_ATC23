import warnings
warnings.filterwarnings("ignore")
import argparse, time
import torch
import torch.nn.functional as F
import os
from dataset import *

from dgl.data import register_data_args
from gcn import GCN
from agnn import AGNN

parser = argparse.ArgumentParser()
register_data_args(parser)
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--dim", type=int, default=96, help="dim")
parser.add_argument("--num_classes", type=int, default=22, help="num_classes")
parser.add_argument("--n-epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
parser.add_argument("--n-layers", type=int, default=2, help="number of layers")
parser.add_argument("--model", type=str, default='gcn', choices=['gcn', 'agnn'], help="type of model")
args = parser.parse_args()
print(args)


def main(args):

    # path = os.path.join("/home/yuke/.graphs/orig", args.dataset)
    path = os.path.join("/home/yuke/.graphs/osdi-ae-graphs", args.dataset + ".npz")
    data = TCGNN_dataset(path, args.dim, args.num_classes, load_from_txt=False)

    g = data.g.int().to(args.gpu)
    features = data.x
    labels = data.y
    in_feats = features.size(1)
    n_classes = data.num_classes

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5).cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    if args.model == 'gcn':
        model = GCN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu)

    if args.model == 'agnn':
        model = AGNN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    ##########################
    # Training GNN model.
    ##########################
    model = model.cuda()
    model.train()

    # dry run.
    for _ in range(3):
         model(features)
    
    torch.cuda.synchronize() 
    t0 = time.perf_counter()

    for _ in range(1, args.n_epochs + 1):
        logits = model(features)
        loss = loss_fcn(logits[:], labels[:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() 
    dur = time.perf_counter() - t0
    print("Time(ms) {:.3f}". format(dur*1e3/args.n_epochs))

if __name__ == '__main__':
    main(args)