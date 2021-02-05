#!/usr/bin/env python3
import torch
import torch.nn as nn
import GAcc
import sys

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new


class GAccFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # SDDMM: edge feature computation. 
        # edge_feature = GAcc.forward_ef(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # GEMM node update
        X_prime = torch.mm(X, weights)
        
        # X_prime_t = torch.ones_like(X_prime)
        # X_prime_t = gen_test_tensor(X_prime)
        # print("=========Before Aggregation========")
        # print(X_prime_t)
        # sys.exit(0)

        # SpMM: Neighbor Aggregation.
        X_prime = GAcc.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SPMM backward propagation.
        d_input_prime = GAcc.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # GEMM backward propagation.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        return d_input, d_weights, None, None, None, None, None, None

class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return GAccFunction.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)


class GAccFunction_GIN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, degrees, partPtr, part2Node, threadPerBlock):
        ctx.threadPerBlock = threadPerBlock

        X = GAcc.forward(X, row_pointers, column_index, degrees, partPtr, part2Node, threadPerBlock)[0]
        ctx.save_for_backward(X, row_pointers, column_index, weights, degrees, partPtr, part2Node)
        X_prime = torch.mm(X, weights)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, row_pointers, column_index, weights, degrees, partPtr, part2Node = ctx.saved_tensors

        d_weights = torch.mm(X.transpose(0,1), d_output)
        d_input_prime = torch.mm(d_output, weights.transpose(0,1))
        d_input = GAcc.backward(d_input_prime, row_pointers, column_index, degrees, partPtr, part2Node, ctx.threadPerBlock)[0]
        
        return d_input, d_weights, None, None, None, None, None, None

class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, X, row_pointers=None, column_index=None, degrees=None, partPtr=None, part2Node=None, threadPerBlock=None):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return GAccFunction_GIN.apply(X, self.weights, row_pointers, column_index, degrees, partPtr, part2Node, threadPerBlock)
