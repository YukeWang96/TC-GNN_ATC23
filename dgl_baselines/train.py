import warnings
warnings.filterwarnings("ignore")

import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import sys
import os
from dataset import *

from gcn import GCN
# from gcn_mp import GCN

# from gcn import GAT
from gat import GAT
from gin import GIN

# run GCN or GAT
def_GCN = True  
def_GIN = False
def_GAT = False

assert sum([def_GCN, def_GIN, def_GAT]) == 1

# Training or Inference
TRAIN = True    

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == "reddit" or args.dataset == "Reddit":
        data = RedditDataset()
    elif args.dataset.startswith('ogb'):
        data = DglNodePropPredDataset(name=args.dataset)
    else:
        path = os.path.join("/home/yuke/.graphs/orig", args.dataset)
        data = GAcc_dataset(path, args.dim, args.num_classes)
    # else:
        # raise ValueError('Unknown dataset: {}'.format(args.dataset))

    # print(path)
    
    ##################
    # For ogb datasets
    ##################
    if args.dataset.startswith('ogb'):
        g, labels = data[0]
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        
        if args.gpu < 0:
            cuda = False
        else:
            cuda = True

        g = g.int().to(args.gpu)
        labels = labels.to(args.gpu)
        # print(labels)
        labels = torch.transpose(labels, 0, 1)

        # print(g)
        # print(g.ndata['feat'])
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        # print(train_idx.size())
        # print(val_idx.size())
        # print(test_idx.size())

        # add self-loop
        print(f"Total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")

        all_nodes = len(train_idx) + len(val_idx) + len(test_idx)
        train_mask = torch.LongTensor(np.ones(all_nodes)).to(args.gpu)
        val_mask = torch.LongTensor(np.ones(all_nodes)).to(args.gpu)
        test_mask = torch.LongTensor(np.ones(all_nodes)).to(args.gpu)
        features = g.ndata['feat']

        in_feats = features.shape[1]
        n_classes = (labels.max() + 1).item()
        n_edges = len(g.all_edges())
        # onehot = torch.zeros([feat.shape[0], n_classes]).to(device)
        # onehot[train_mask, labels[train_mask, 0]] = 1
        # return torch.cat([feat, onehot], dim=-1)
        # labels = onehot

    ##########################
    # For DGL-builtin datasets
    ##########################
    elif args.dataset in ['cora', 'citeseer', 'pubmed', 'reddit']:
        g = data[0]
        
        if args.gpu < 0:
            cuda = False
        else:
            cuda = True

        # print(g.edges())
        # with open("citeseer", "w") as fp:
            # for src, trg in zip(g.edges()[0], g.edges()[1]):
                # fp.write(f"{src} {trg}\n")
        # fp.close()
        # sys.exit(0)

        g = g.int().to(args.gpu)
        features = g.ndata['feat']
        labels = g.ndata['label']
        # print(labels)
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
    
    ##########################
    # For all other datasets
    ##########################
    else:
        g = data.g

        if args.gpu < 0:
            cuda = False
        else:
            cuda = True

        g = g.int().to(args.gpu)

        features = data.x
        labels = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        in_feats = features.size(1)
        n_classes = data.num_classes
        n_edges = data.num_edges

    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #           train_mask.int().sum().item(),
    #           val_mask.int().sum().item(),
    #           test_mask.int().sum().item()))

    
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    if def_GCN:
        print("Run GCN")
        model = GCN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)
    if def_GAT:
        print("Run GAT")
        model = GAT(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)
    
    if def_GIN:
        print("Run GIN")
        model = GIN(g,
                    in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu,
                    args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    ##########################
    # Training GNN model.
    ##########################
    dur = []
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        if epoch >= 3: t0 = time.time()

        # forward
        logits = model(features)

        if TRAIN:
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize() 
        
        if epoch >= 3: dur.append(time.time() - t0)
        if epoch == args.n_epochs:
            print("Epoch {:05d} | Time(ms) {:.3f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur) * 1e3, n_edges / np.mean(dur) / 1000))

        # acc = evaluate(model, features, labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
        #                                      acc, n_edges / np.mean(dur) / 1000))

    # print()
    # acc = evaluate(model, features, labels, test_mask)
    # print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dim", type=int, default=96, help="dim")
    parser.add_argument("--num_classes", type=int, default=22, help="num_classes")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)
    args = parser.parse_args()
    # print(args)
    main(args)