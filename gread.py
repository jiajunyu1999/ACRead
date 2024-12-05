import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import reset 
from torch_scatter import scatter, scatter_add, scatter_max, scatter_min
from torch_geometric.nn import MLPAggregation, GRUAggregation, DeepSetsAggregation, Linear, SetTransformerAggregation
from torch_geometric.utils import degree
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(MLP, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(), 
            # nn.BatchNorm1d(hid_dim),            
            torch.nn.Linear(256, hid_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, score_dim, embed_dim, args):
        super().__init__()
        self.num_heads = num_heads
        self.args = args
        self.scale_factor = nn.Parameter(torch.ones(1))  # Learnable scaling factor

        self.q = nn.ModuleList([nn.Linear(score_dim, embed_dim) for _ in range(num_heads)])
        self.k = nn.ModuleList([nn.Linear(score_dim, embed_dim) for _ in range(num_heads)])
        self.v = nn.ModuleList([nn.Linear(score_dim, embed_dim) for _ in range(num_heads)])
        self.out = nn.Linear(embed_dim*num_heads, embed_dim)

        
        self.outs = nn.ModuleList([nn.Linear(embed_dim * num_heads, embed_dim) for _ in range(num_heads)])
        self.gate = nn.Linear(score_dim, 1)
        self.pos = nn.Parameter(torch.randn(1, score_dim))
        self.norm = nn.BatchNorm1d(embed_dim)
        self.para = nn.Parameter(torch.randn(1, embed_dim))
    
    def forward(self, score_matrix, x, batch):
        score_matrix = score_matrix + self.pos
        q = torch.cat([head(score_matrix).unsqueeze(1) for head in self.q], dim=1)   # b x hd
        k = torch.cat([head(score_matrix).unsqueeze(1) for head in self.k], dim=1)
        v = torch.cat([head(score_matrix).unsqueeze(1) for head in self.v], dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(k.size(-1)))
        attention_weights = softmax(scores, batch.batch) 
        # attention_weights = F.dropout(attention_weights, p=self.args.drop)
        s = torch.matmul(attention_weights, v).view(score_matrix.size(0), -1)
        s = self.out(s)
        return s


class ACRead(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, num_edgefeats, args, score_nums = 10):
        super(ACRead, self).__init__()

        self.hid_dim = hid_dim
        self.args = args
        self.score_nums = score_nums

        self.edge_encoder = MLP(num_edgefeats, self.hid_dim)   
        self.init_x_encoder_mlp = MLP(in_dim, hid_dim)

        self.att_encoder = MultiHeadAttention(self.args.head, self.score_nums, 128, args)
        
        self.score_norm = nn.BatchNorm1d(self.score_nums)
        self.x_norm = nn.BatchNorm1d(self.hid_dim)
        self.score_lin = nn.Linear(self.score_nums, self.score_nums, bias=False)
        
        nn.init.ones_(self.score_lin.weight/self.args.score_nums)
        self.x_batch_norms = torch.nn.ModuleList()
        for i in range(self.args.num_layer):
            self.x_batch_norms.append(nn.BatchNorm1d(self.hid_dim*2))
        self.weight = nn.Linear(self.score_nums, 1)
        
        nn.init.constant_(self.weight.weight, 1/self.score_nums)

        self.mlp = MLP(self.hid_dim, hid_dim)

        self.atom_encoder = AtomEncoder(emb_dim = in_dim)
        self.bond_encoder = BondEncoder(emb_dim = num_edgefeats)

        self.s_lin = nn.Linear(self.hid_dim, 1)
        self.map_layer = nn.Linear(128, self.hid_dim)
        self.fusion_batch_norm = nn.BatchNorm1d(self.hid_dim)
        self.fusion_layer1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fusion_layer2 = nn.Linear(self.hid_dim, self.hid_dim)
        

    def agg_edge(self, x, edge_index, edge_embedding, norm):
        src, dst = edge_index
        edge_score = (x[src] * edge_embedding).sum(dim=-1, keepdim=True)
        attention_weights = softmax(edge_score, src)
        new_x = torch.zeros_like(x)
        new_x.index_add_(0, src, attention_weights * x[src])  

        edge_score = (x[dst] * edge_embedding).sum(dim=-1, keepdim=True)
        attention_weights = softmax(edge_score, dst)
        new_x2 = torch.zeros_like(x)
        new_x2.index_add_(0, dst, attention_weights * x[dst])  

        return (new_x2+new_x)/2

    def forward(self, h, batch, size=None):
        init_x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        if init_x is None:
            init_x = torch.ones_like(batch.batch, dtype=torch.float).unsqueeze(dim=1).repeat(1,10)
        if edge_attr is None:
            edge_attr = torch.ones_like(edge_index.transpose(0, 1), dtype=torch.float)[:, 0:1]
        if self.args.dataset.count('ogbg') != 0:
            init_x = self.atom_encoder(init_x)
            edge_attr = self.bond_encoder(edge_attr)
        x = self.init_x_encoder_mlp(init_x)
        edge_embedding = self.edge_encoder(edge_attr)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1

        score_matrix = batch.centrality[:,0:self.score_nums]        
        score_matrix = self.score_lin(score_matrix)
        score_matrix = self.score_norm(score_matrix)
        s = self.att_encoder(score_matrix, x, batch)
        s = self.map_layer(s)
        s = softmax(s, batch.batch)
        if self.args.phi == 'hadama':
            h = s * h
        elif self.args.phi == 'adding':
            h = s + h
        elif self.args.phi == 'gated':
            gated = softmax(self.s_lin(s), batch.batch)
            h = gated * s * h
        elif self.args.phi == 'linear':
            h = self.s_lin(s) * h
        elif self.args.phi == 'fusion':
            h = self.fusion_layer1(s) + self.fusion_layer2(h)
        h = scatter(h, batch.batch, dim=0, dim_size=size, reduce='sum')
        return h



def global_add_pool(x, batch, size = None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='add')


def global_mean_pool(x, batch, size = None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def global_max_pool(x, batch, size = None):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='max')


def scale(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)
    sc_max = scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    sc_min = scatter_min(src, index, dim=0, dim_size=num_nodes)[0][index]
    
    out = (src - sc_max) / (sc_max - sc_min + 1e12)


    return out


def softmax(src, index, num_nodes=None):
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


class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch.max().item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()


    def forward(self, x, batch, size=None):
        """"""
        batch_size = batch.max().item() + 1 if size is None else size

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GlobalReadout(torch.nn.Module):
    def __init__(self, num_nodefeats, num_edgefeats, emb_dim = 300, read_op = 'sum', args = None, score_nums = 4):

        super(GlobalReadout, self).__init__()

        self.emb_dim = emb_dim
        self.read_op = read_op

        if read_op == 'sum':
            self.gread = global_add_pool
        elif read_op == 'mean':
            self.gread = global_mean_pool
        elif read_op == 'max':
            self.gread = global_max_pool
        elif read_op == 'attention':
            self.gread = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif read_op == 'set2set':
            self.gread = Set2Set(emb_dim, processing_steps = 2)
        elif read_op == 'gru':
            self.gread = GRUAggregation(self.emb_dim, self.emb_dim)
        elif read_op == 'mlp':
            self.gread = MLPAggregation(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                max_num_elements=3,
                num_layers=1,
            )
        elif read_op == 'st':
            self.gread = SetTransformerAggregation(
                channels=self.emb_dim, heads=2)
            self.gread.reset_parameters()
        elif read_op == 'deepset':
            self.gread = DeepSetsAggregation(
                local_nn=Linear(self.emb_dim, self.emb_dim),
                global_nn=Linear(self.emb_dim, self.emb_dim),
            )
        elif read_op == 'acread':
            self.gread = ACRead(num_nodefeats, emb_dim, num_edgefeats, args, score_nums = score_nums)
        
        
    def forward(self, x, batch):
        if self.read_op == 'acread':
            return self.gread(x, batch)
        else:
            return self.gread(x, batch.batch)
