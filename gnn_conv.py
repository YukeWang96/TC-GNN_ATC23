#!/usr/bin/env python3
import torch
import sys
import math
import time 

from tqdm.std import tqdm
import TCGNN

n_heads = 8
n_output = 8

def gen_test_tensor(X_prime):
    n_rows = X_prime.size(0)
    n_cols = X_prime.size(1)
    
    X_new = []
    for i in range(n_rows):
        tmp = [i] * n_cols
        X_new.append(tmp)

    X_new = torch.FloatTensor(X_new).cuda()
    return X_new


class TCGNNFunction_SAG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, row_pointers, column_index, \
                blockPartition, edgeToColumn, edgeToRow):

        ctx.save_for_backward(row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow)

        # Basic Scatter and Gather
        X_out = TCGNN.forward(X, row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow)[0]

        return X_out

    @staticmethod
    def backward(ctx, d_output):
        row_pointers, column_index, \
            blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SAG backward.
        d_input = TCGNN.forward(d_output, row_pointers, column_index, \
                                blockPartition, edgeToColumn, edgeToRow)[0]

        return d_input, None, None, None, None, None, None


class TCGNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        # X = torch.sparse.mm(edge_coo, X)
        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # GEMM node update
        X_prime = torch.mm(X, weights)
        
        # X_prime_t = torch.ones_like(X_prime)
        # X_prime_t = gen_test_tensor(X_prime)
        # print("=========Before AggreAGNNion========")
        # print(X_prime_t)
        # sys.exit(0)

        # SpMM: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        # print("==========After Aggreation=========")
        # print(X_prime)
        # sys.exit(0)

        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SPMM backward propaAGNNion.
        d_input_prime = TCGNN.forward(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)
        return d_input, d_weights, None, None, None, None, None, None

class TCGNNFunction_GIN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):

        # SpMM: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward(X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        
        ctx.save_for_backward(X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # GEMM node update
        X_prime = torch.mm(X_prime, weights)
        
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X_prime, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # GEMM backward propaAGNNion.
        d_X_prime = torch.mm(d_output, weights.transpose(0,1))
        d_weights = torch.mm(X_prime.transpose(0,1), d_output)

        # SPMM backward propaAGNNion.
        d_input = TCGNN.forward(d_X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        return d_input, d_weights, None, None, None, None, None, None
        # return None, d_weights, None, None, None, None, None, None

class TCGNNFunction_AGNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):

        ctx.save_for_backward(X, weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

        # GEMM node update
        X_prime = torch.mm(X, weights)
        
        # SDDMM: edge feature computation. 
        edge_feature = TCGNN.forward_ef(X_prime, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]

        # Edge Attention Generation: [n_e, n_head]       
        edge_attentions = torch.mm(edge_feature.unsqueeze(-1), attention_w).transpose(0,1).contiguous()
        # print(edge_attentions.size())

        # SpMM_AGNN: Neighbor AggreAGNNion.
        X_prime = TCGNN.forward_AGNN(X_prime, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow)[0]

        ctx.save_for_backward(X, weights, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow)
        # print("==========After Aggreation=========")
        return X_prime

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow = ctx.saved_tensors

        # SPMM backward propaAGNNion.
        d_input_prime = TCGNN.forward_AGNN(d_output, row_pointers, column_index, edge_attentions, blockPartition, edgeToColumn, edgeToRow)[0]

        # GEMM backward propaAGNNion.
        d_input = torch.mm(d_input_prime, weights.transpose(0,1))
        d_weights = torch.mm(X.transpose(0,1), d_input_prime)

        # attention weight back propaAGNNion.
        d_attention = TCGNN.forward_ef(d_output, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)[0]
        # print(d_attention.size())
        d_attention_exp = d_attention[None, :].expand(8, -1)
        # print(d_attention_exp.size())

        d_attention_w = torch.mm(d_attention_exp, column_index[:, None].float()).transpose(0,1)
        # print(d_attention_w.size())

        return d_input, d_weights, d_attention_w, None, None, None, None, None





###################################
# Definition of each conv layers
###################################
class SAG(torch.nn.Module):
    def __init__(self, row_pointers, column_index, \
                    blockPartition, edgeToColumn, edgeToRow):
        super(SAG, self).__init__()

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.blockPartition = blockPartition
        self.edgeToColumn = edgeToColumn
        self.edgeToRow = edgeToRow


    def profile(self, X, num_rounds=1):
        
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in tqdm(range(num_rounds)):
            TCGNNFunction_SAG.apply(X, self.row_pointers, self.column_index, \
                                        self.blockPartition, self.edgeToColumn, self.edgeToRow)
        torch.cuda.synchronize()
        dur = time.perf_counter() - start
        # print("=> SAG profiling avg (ms): {:.3f}".format(dur*1e3/num_rounds))
        # print()

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
        return TCGNNFunction.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)

class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, X, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow):
        '''
        @param:
        X:  the input tensor of the graph node embedding, shape: [n_nodes, n_dim].
        A:  the CSR node pointer of the graph, shape: [node, 1].
        edges: the CSR edge list of the graph, shape: [edge, 1].
        partitioin: for the graph with the part-based optimziation.
        '''
        return TCGNNFunction_GIN.apply(X, self.weights, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)


class AGNNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AGNNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.attention_w = torch.nn.Parameter(torch.randn(1, n_heads))
        self.reset_parameters()

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
        return TCGNNFunction_AGNN.apply(X, self.weights, self.attention_w, row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow)
