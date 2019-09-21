import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DG_fc(nn.Module):
    def __init__(self,infeature,hidden,heads,layers,dropout,layer_norm):
        super().__init__()
        #self.embed = nn.Embedding(100,hidden)
        self.mlp = nn.Linear(infeature,hidden)
        self.dynamic_gnns = nn.ModuleList([DynamicGNN(hidden, heads, dropout,layer_norm=layer_norm)\
                                           for _ in range(layers)])

        # h = int(62*200/infeature*hidden)

        self.linear = nn.Linear(hidden,hidden)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden,3)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x,adj):
        # x:(b,n,L,1)
        x = self.mlp(x) # (b,n,L,h)
        for dgnn in self.dynamic_gnns:
            x = dgnn(x,adj)

        x = torch.sum(torch.sum(x,dim=1),dim=1)
        # x = x.contiguous().view(x.size(0),-1)
        out = self.out(self.tanh(self.linear(x)))# (b,3)
        out = self.softmax(out)
        return out

class GAT(nn.Module):
    def __init__(self,infeature,hidden,heads,layers,dropout):
        super().__init__()
        self.mlp = nn.Linear(infeature, hidden)
        self.gats = nn.ModuleList([GATlayer(hidden,heads,dropout) for _ in range(layers)])
        self.linear = nn.Linear(hidden,hidden)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden,3)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.bn = nn.BatchNorm1d(62)
    def forward(self, x,adj):
        # x (b,n,L)
        x = self.bn(x)
        x = self.mlp(x)
        for gat in self.gats:
            x = gat(x,adj)

        x = torch.sum(x,dim=1)
        #out = self.out(self.tanh(self.linear(x)))# (b,3)
        out = self.out(x)
        out = self.softmax(out)
        return out



class DynamicGNN(nn.Module):
    def __init__(self,hidden, heads, dropout=0.1, alpha=0.2,layer_norm=False):
        super().__init__()
        self.att = MultiHeadedAttention(hidden,heads)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.resln = ResLN(size=hidden,dropout=dropout)
        self.gat = GATlayer(hidden,heads,dropout,alpha)
        self.bn = nn.BatchNorm2d(62)
    def forward(self, x,adj):
        #x (b,n,l,h)
        x = self.bn(x)
        b,n,l,h = x.size()


        # time attention
        x = x.contiguous().view(-1,l,h)
        if self.layer_norm:
            x = self.resln.forward(x, lambda a:self.att(a,a,a))
        else:
            x = self.att.forward(x,x,x)
        x = x.contiguous().view(b,n,l,h)
        x = self.bn(x)

        # space attention
        x = x.permute(0,2,1,3).contiguous().view(b*l,n,h)
        new_adj = adj.repeat(1,l,1).view(b*l,n,n)
        x = self.gat(x,new_adj)
        x = x.view(b,l,n,h).permute(0,2,1,3)
        return x

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden, head, dropout=0.1):
        super().__init__()
        assert hidden % head == 0

        # We assume d_v always equals d_k
        self.d_k = hidden // head
        self.h = head

        self.linear_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output_linear = nn.Linear(hidden, hidden)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class GATlayer(nn.Module):
    def __init__(self, nhid, nheads,  dropout=0.1, alpha=0.2):
        """Dense version of GAT."""
        super(GATlayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.attentions = nn.ModuleList([GraphAttentionLayer(nhid, int(nhid/nheads), dropout=dropout,
            alpha=alpha, concat=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        return self.dropout(x)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # bs,N,d; bs,N,N
        # print(input.get_device(),self.W.get_device())
        h = torch.matmul(input, self.W)  # bs,N,d1
        N = h.size()[1]
        batch_size = h.size()[0]

        # bs,N*N,2*out
        a_input = torch.cat([h.repeat(1, N, 1), h.repeat(1, 1, N).view(batch_size, N * N, -1)], dim=2)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)).view(batch_size, N, N)

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # bs,N,N
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # bs,N,d
        if self.concat:
            return F.elu(h_prime)  # bs,N,N
        else:
            return h_prime  # bs,N,N

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class ResLN(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResLN, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2