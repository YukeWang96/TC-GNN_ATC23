#!/usr/bin/env python3
from numpy.core import numeric
import torch
import numpy as np
import time

from config import *
from scipy.sparse import *

class TCGNN_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, dim, num_class, load_from_txt=True, verbose=False, sparsity=1):
        super(TCGNN_dataset, self).__init__()

        self.nodes = set()

        self.load_from_txt = load_from_txt
        self.num_nodes = 0
        self.num_features = dim 
        self.num_classes = num_class
        self.edge_index = None
        
        self.reorder_flag = False
        self.verbose_flag = verbose
        self.sparsity = sparsity

        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.syn = False

        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)

        train = 1
        val = 0.3
        test = 0.1
        self.train_mask = [1] * int(self.num_nodes * train) + [0] * (self.num_nodes  - int(self.num_nodes * train))
        self.val_mask = [1] * int(self.num_nodes * val)+ [0] * (self.num_nodes  - int(self.num_nodes * val))
        self.test_mask = [1] * int(self.num_nodes * test) + [0] * (self.num_nodes  - int(self.num_nodes * test))
        self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):

        if self.syn:
            self.num_nodes = 3000
            node_li = list(range(self.num_nodes))
            from itertools import product
            src_li = []
            dst_li = []
            cnt = 0
            for src, dst in product(node_li, node_li):
                src_li.append(src)
                dst_li.append(dst)
                cnt += 1
            self.edge_index = np.stack([src_li, dst_li])

        else:
            # loading from a txt graph file
            if self.load_from_txt:
                fp = open(path, "r")
                src_li = []
                dst_li = []
                start = time.perf_counter()
                for line in fp:
                    src, dst = line.strip('\n').split()
                    src, dst = int(src), int(dst)
                    src_li.append(src)
                    dst_li.append(dst)
                    self.nodes.add(src)
                    self.nodes.add(dst)
                
                self.num_edges = len(src_li)
                self.num_nodes = max(self.nodes) + 1
                self.edge_index = np.stack([src_li, dst_li])

                dur = time.perf_counter() - start
                if self.verbose_flag:
                    print("# Loading (txt) {:.3f}s ".format(dur))

            # loading from a .npz graph file
            else: 
                if not path.endswith('.npz'):
                    raise ValueError("graph file must be a .npz file")

                start = time.perf_counter()
                graph_obj = np.load(path)
                src_li = graph_obj['src_li']
                dst_li = graph_obj['dst_li']

                self.num_nodes = graph_obj['num_nodes']
                self.num_edges = len(src_li)
                self.edge_index = np.stack([src_li, dst_li])
                dur = time.perf_counter() - start
                if self.verbose_flag:
                    print("# Loading (npz)(s): {:.3f}".format(dur))
        
        self.num_edges = num_edges = len(self.edge_index[0])
        self.avg_degree = self.num_edges / self.num_nodes
        self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))

        if self.verbose_flag:
            print('# nodes: {}'.format(self.num_nodes))
            print("# avg_degree: {:.2f}".format(self.avg_degree))
            print("# avg_edgeSpan: {}".format(int(self.avg_edgeSpan)))

        # sparsity = 0.5
        # print("Before Drop Edges: ", num_edges)
        # n = int(self.sparsity * num_edges) 
        # index = np.random.choice(num_edges, n, replace=False)  
        # print("After Select Edges: {}", len(index))
        # self.num_edges = len(index)

        # src = self.edge_index[0][index]
        # dst = self.edge_index[1][index]
        # print(len(src))
        # print(len(dst))
        # self.edge_index = np.stack([src, dst])
        # print(self.edge_index)
        # exit(0)

        # Build graph CSR.
        val = [1] * len(self.edge_index[0])
        start = time.perf_counter()
        scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        scipy_csr = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start

        if self.verbose_flag:
            print("# Build CSR (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)

        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    def init_labels(self, num_class):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.ones(self.num_nodes).long().cuda()
