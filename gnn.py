import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
from conv import *
from gread import GlobalReadout
from sread import SemanticReadout
from torch_geometric.utils import to_dense_adj
from torch import Tensor
# from kernel import Kernel_readout

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import reset 
from torch_scatter import scatter, scatter_add, scatter_max

class GNN(torch.nn.Module):

    def __init__(self, gnn_type, num_classes, num_nodefeats, num_edgefeats,
                    num_layer = 5, emb_dim = 300, drop_ratio = 0.5,  
                    read_op = 'sum', num_centers = 4, args = None):
        super(GNN, self).__init__()

        self.num_classes = num_classes
        self.num_nodefeats = num_nodefeats
        self.num_edgefeats = num_edgefeats        

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.rep_dim = emb_dim * num_centers 
        self.args = args

        self.read_op = read_op
        self.num_centers = num_centers
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_nodefeats, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim))

        
        ### GNN to generate node-level representations
        self.gnn_node = GNN_node(num_nodefeats, num_edgefeats, num_layer, emb_dim, drop_ratio = drop_ratio, gnn_type = gnn_type)
        
        ### Readout layer to generate graph-level representations
        if read_op == 'sread':
            self.read_op = SemanticReadout(self.emb_dim, read_op=self.read_op, num_position=self.num_centers)
        else:
            self.read_op = GlobalReadout(self.num_nodefeats, num_edgefeats, self.emb_dim, read_op=read_op, args = args, score_nums = args.score_nums)
        
        if read_op == 'set2set': self.emb_dim *= 2
        if read_op == 'sread': self.emb_dim *= 4
        

        self.graph_mlp_pred = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.emb_dim),
                                                  torch.nn.ReLU(), torch.nn.Linear(self.emb_dim, self.num_classes))
        self.graph_lin_pred = torch.nn.Linear(self.emb_dim, self.num_classes)
        self.lin = torch.nn.Linear(self.num_nodefeats, emb_dim)
        self.mlp =  torch.nn.Sequential(torch.nn.Linear(self.num_nodefeats, self.emb_dim),
                                                  torch.nn.ReLU(), torch.nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, batched_data):
        # if self.args.read_op != 'weight_sum':
        if self.args.gnn == 'x':
            h_node = batched_data.x
        elif self.args.gnn == 'lin':
            h_node = self.lin(batched_data.x)
        elif self.args.gnn == 'mlp':
            h_node = self.mlp(batched_data.x)
        else:
            h_node = self.gnn_node(batched_data)
        h_graph = self.read_op(h_node, batched_data)   # 1*d
        return self.graph_mlp_pred(h_graph)

    def get_node_embed(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return F.normalize(h_node)

    def get_alignment(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return self.read_op.get_alignment(h_node)

    def get_aligncost(self, batched_data):
        h_node = self.node_encoder(batched_data.filter_x)
        return self.read_op.get_aligncost(h_node, batched_data.batch)

    def softmax(self, src, index, num_nodes=None):
        r"""Computes a sparsely evaluated softmax.
        Given a value tensor :attr:`src`, this function first groups the values
        along the first dimension based on the indices specified in :attr:`index`,
        and then proceeds to compute the softmax individually for each group.

        Args:
            src (Tensor): The source tensor.
            index (LongTensor): The indices of elements for applying the softmax.
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

        :rtype: :class:`Tensor`
        """

        num_nodes = maybe_num_nodes(index, num_nodes)

        out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
        out = out.exp()
        out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

        return out

if __name__ == '__main__':
    GNN(num_classes = 10)
