import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from utils import sample_neighbors
import queue


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), dim=-1))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau.unsqueeze(-1), w) + b, arg)
    else:
        v1 = f(torch.matmul(tau.unsqueeze(-1), w) + b)
    v2 = torch.matmul(tau.unsqueeze(-1), w0) + b0
    return torch.cat([v1, v2], dim=-1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.device = device
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src_mask = self.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        src = torch.transpose(src, 1, 0)
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        x = torch.transpose(x, 1, 0)
        out_poi = self.decoder_poi(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_cat


class TimeAwareTransformer(nn.Module):
    def __init__(self, num_poi, num_cat, nhid, batch_size, device, dropout):
        super(TimeAwareTransformer, self).__init__()

        self.device = device
        self.nhid = nhid
        self.batch_size = batch_size
        # self.encoder = nn.Embedding(num_poi, embed_size)

        self.decoder_poi = nn.Linear(nhid, num_poi)
        self.tu = 24 * 3600
        self.time_bin = 3600
        assert (self.tu) % self.time_bin == 0
        self.day_embedding = nn.Embedding(8, nhid, padding_idx=0)
        self.hour_embedding = nn.Embedding(int((self.tu) / self.time_bin) + 2, nhid, padding_idx=0)

        self.W1_Q = nn.Linear(nhid, nhid)
        self.W1_K = nn.Linear(nhid, nhid)
        self.W1_V = nn.Linear(nhid, nhid)
        self.norm11 = nn.LayerNorm(nhid)
        self.feedforward1 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid)
        )
        self.norm12 = nn.LayerNorm(nhid)

        self.W2_Q = nn.Linear(nhid, nhid)
        self.W2_K = nn.Linear(nhid, nhid)
        self.W2_V = nn.Linear(nhid, nhid)
        self.norm21 = nn.LayerNorm(nhid)
        self.feedforward2 = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid)
        )
        self.norm22 = nn.LayerNorm(nhid)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, batch_seq_lens, batch_input_seqs_ts, batch_label_seqs_ts):
        hourInterval = torch.zeros((src.shape[0], src.shape[1], src.shape[1]), dtype=torch.long).to(self.device)
        dayInterval = torch.zeros((src.shape[0], src.shape[1], src.shape[1]), dtype=torch.long).to(self.device)

        label_hourInterval = torch.zeros((src.shape[0], src.shape[1], src.shape[1]), dtype=torch.long).to(self.device)
        label_dayInterval = torch.zeros((src.shape[0], src.shape[1], src.shape[1]), dtype=torch.long).to(self.device)
        for i in range(src.shape[0]):
            for j in range(batch_seq_lens[i]):
                for k in range(j + 1):
                    if i == j:
                        hourInterval[i][j][k] = 1
                    else:
                        hourInterval[i][j][k] = int(
                            ((batch_input_seqs_ts[i][j] - batch_input_seqs_ts[i][k]) % (self.tu)) / self.time_bin) + 2
                    dayInterval[i][j][k] = int((batch_input_seqs_ts[i][j] - batch_input_seqs_ts[i][k]) / (self.tu)) + 1
                    if dayInterval[i][j][k] > 6:
                        dayInterval[i][j][k] = 7
                    label_hourInterval[i][j][k] = int(
                        ((batch_label_seqs_ts[i][j] - batch_input_seqs_ts[i][k]) % (self.tu)) / self.time_bin) + 2
                    label_dayInterval[i][j][k] = int(
                        (batch_label_seqs_ts[i][j] - batch_input_seqs_ts[i][k]) / (self.tu)) + 1
                    if label_dayInterval[i][j][k] > 6:
                        label_dayInterval[i][j][k] = 7

        hourInterval_embedding = self.hour_embedding(hourInterval)
        dayInterval_embedding = self.day_embedding(dayInterval)

        label_hourInterval_embedding = self.hour_embedding(label_hourInterval)
        label_dayInterval_embedding = self.day_embedding(label_dayInterval)

        # mask attn
        attn_mask = ~torch.tril(torch.ones((src.shape[1], src.shape[1]), dtype=torch.bool, device=self.device))
        time_mask = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool, device=self.device)
        for i in range(src.shape[0]):
            time_mask[i, batch_seq_lens[i]:] = True

        attn_mask = attn_mask.unsqueeze(0).expand(src.shape[0], -1, -1)
        time_mask = time_mask.unsqueeze(-1).expand(-1, -1, src.shape[1])

        Q = self.W1_Q(src)
        K = self.W1_K(src)
        V = self.W1_V(src)

        attn_weight = Q.matmul(torch.transpose(K, 1, 2))
        attn_weight += hourInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight += dayInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)

        attn_weight = attn_weight / math.sqrt(self.nhid)

        paddings = torch.ones(attn_weight.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)

        attn_weight = torch.where(time_mask, paddings, attn_weight)
        attn_weight = torch.where(attn_mask, paddings, attn_weight)

        attn_weight = F.softmax(attn_weight, dim=-1)
        x = attn_weight.matmul(V)  # B,L,D
        x += torch.matmul(attn_weight.unsqueeze(2), hourInterval_embedding).squeeze(2)
        x += torch.matmul(attn_weight.unsqueeze(2), dayInterval_embedding).squeeze(2)

        x = self.norm11(x + src)
        ffn_output = self.feedforward1(x)
        ffn_output = self.norm12(x + ffn_output)
        '''

        src=ffn_output

        Q = self.W2_Q(src)
        K = self.W2_K(src)
        V = self.W2_V(src)

        attn_weight = Q.matmul(torch.transpose(K, 1, 2))
        attn_weight += hourInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight += dayInterval_embedding.matmul(Q.unsqueeze(-1)).squeeze(-1)
        attn_weight = attn_weight / math.sqrt(self.nhid)
        paddings = torch.ones(attn_weight.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)

        attn_weight = torch.where(time_mask, paddings, attn_weight)
        attn_weight = torch.where(attn_mask, paddings, attn_weight)

        attn_weight = F.softmax(attn_weight, dim=-1)
        x = attn_weight.matmul(V)  # B,L,D
        x += torch.matmul(attn_weight.unsqueeze(2), hourInterval_embedding).squeeze(2)
        x += torch.matmul(attn_weight.unsqueeze(2), dayInterval_embedding).squeeze(2)

        x = self.norm21(x + src)
        ffn_output = self.feedforward2(x)
        ffn_output = self.norm22(x + ffn_output)
        '''

        # attn_mask=attn_mask.unsqueeze(-1).expand(-1,-1,-1,ffn_output.shape[-1])
        ffn_output = ffn_output.unsqueeze(2).repeat(1, 1, ffn_output.shape[1], 1).transpose(2, 1)
        ffn_output = torch.add(ffn_output, label_hourInterval_embedding)
        ffn_output = torch.add(ffn_output, label_dayInterval_embedding)
        '''
        paddings = torch.ones(ffn_output.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)
        ffn_output = torch.where(attn_mask, paddings, ffn_output)
        '''
        decoder_output_poi = self.decoder_poi(ffn_output)
        pooled_poi = torch.zeros(decoder_output_poi.shape[0], decoder_output_poi.shape[1],
                                 decoder_output_poi.shape[3]).to(self.device)
        for i in range(decoder_output_poi.shape[1]):
            pooled_poi[:, i] = torch.mean(decoder_output_poi[:, i, :i + 1], dim=1)

        return pooled_poi


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings and transform
    """

    def __init__(self, id2feat, device):
        """
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """
        super(MeanAggregator, self).__init__()
        self.id2feat = id2feat
        self.device = device

    def forward(self, to_neighs):
        """
        nodes --- list of nodes in a batch
        dis --- shape alike adj
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        tmp = [n for x in to_neighs for n in x]
        unique_nodes_list = set(tmp)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(len(to_neighs), len(unique_nodes_list)).to(self.device)
        column_indices = [unique_nodes[n] for n in tmp]
        row_indices = [i for i in range(len(to_neighs)) for j in range(len(to_neighs[i]))]

        mask[row_indices, column_indices] += 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        embed_matrix = self.id2feat(
            torch.LongTensor(list(unique_nodes_list)).to(self.device))  # （unique_count, feat_dim)
        to_feats = mask.mm(embed_matrix)  # n * embed_dim
        return to_feats  # n * embed_dim


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    id2feat -- function mapping LongTensor of node ids to FloatTensor of feature values.
    cuda -- whether to use GPU
    gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
    """

    def __init__(self, id2feat, restart_prob, num_walks, input_dim, output_dim, device, dropout,
                 id, adj_queues, dis_queues, all_adj_queues, all_dis_queues):
        super(SageLayer, self).__init__()
        self.id2feat = id2feat
        self.dis_agg = MeanAggregator(self.id2feat, device)
        self.adj_agg = MeanAggregator(self.id2feat, device)
        self.device = device
        self.adj_list = None
        self.dis_list = None
        self.restart_prob = restart_prob
        self.num_walks = num_walks
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.dropout = dropout
        self.adj_queues = adj_queues
        self.dis_queues = dis_queues
        self.all_adj_queues = all_adj_queues
        self.all_dis_queues = all_dis_queues
        self.id = id
        self.W_self = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.W_adj = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.W_dis = nn.Linear(input_dim, int(output_dim / 3), bias=False)
        self.WC = nn.Linear(output_dim, output_dim)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W_self.weight.data.uniform_(-initrange, initrange)
        self.W_adj.weight.data.uniform_(-initrange, initrange)
        self.W_dis.weight.data.uniform_(-initrange, initrange)
        self.bias.data.zero_()

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """

        unique_nodes_list = list(set([int(node) for node in nodes]))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        adj_neighbors = [[] for _ in unique_nodes_list]
        dis_neighbors = [[] for _ in unique_nodes_list]
        missing_adj_idx = []
        missing_dis_idx = []
        for idx, node in enumerate(unique_nodes_list):
            try:
                if self.training:
                    random_walk = self.adj_queues[node].get_nowait()
                else:
                    random_walk = self.all_adj_queues[node].get_nowait()
                adj_neighbors[idx] = random_walk
            except queue.Empty:
                missing_adj_idx.append(idx)
            try:
                if self.training:
                    random_walk = self.dis_queues[node].get_nowait()
                else:
                    random_walk = self.all_dis_queues[node].get_nowait()
                dis_neighbors[idx] = random_walk
            except queue.Empty:
                missing_dis_idx.append(idx)

        if len(missing_adj_idx) != 0:
            missing_adj_neighbors = sample_neighbors(self.adj_list, [unique_nodes_list[i] for i in missing_adj_idx],
                                                     self.restart_prob, self.num_walks, 'adj')
            for idx, missing_adj_neighbor in zip(missing_adj_idx, missing_adj_neighbors):
                adj_neighbors[idx] = missing_adj_neighbor
        if len(missing_dis_idx) != 0:
            missing_dis_neighbors = sample_neighbors(self.dis_list, [unique_nodes_list[i] for i in missing_dis_idx],
                                                     self.restart_prob, self.num_walks, 'dis')
            for idx, missing_dis_neighbor in zip(missing_dis_idx, missing_dis_neighbors):
                dis_neighbors[idx] = missing_dis_neighbor

        self_feats = self.id2feat(torch.tensor(unique_nodes_list).to(self.device))
        adj_feats = self.adj_agg(adj_neighbors)
        dis_feats = self.dis_agg(dis_neighbors)

        adj_feats = self.W_adj(adj_feats)
        self_feats = self.W_self(self_feats)
        self_feats = F.dropout(self_feats, p=self.dropout, training=self.training)
        dis_feats = self.W_dis(dis_feats)
        feats = torch.cat((self_feats, adj_feats, dis_feats), dim=-1) + self.bias
        feats = self.WC(feats)
        feats = self.leakyRelu(feats)
        feats = F.normalize(feats, p=2, dim=-1)

        nodes_idx = [unique_nodes[int(node)] for node in nodes]
        res = feats[nodes_idx]

        return res

    def set_adj(self, adj, dis):
        self.adj_list = adj
        self.dis_list = dis


class GraphSage(nn.Module):
    def __init__(self, input_dim, embed_dim, device, restart_prob, num_walks, dropout, adj_queues, dis_queues,
                 all_adj_queues, all_dis_queues):
        super(GraphSage, self).__init__()
        self.id2node = None
        self.device = device
        '''
        self.layer1 = SageLayer(id2feat=lambda nodes: self.id2node[nodes], adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=X.shape[1],output_dim=embed_dim, device=device,
                                dropout=dropout,workers=workers,pool=self.pool)
        self.layer2 = SageLayer(id2feat=lambda nodes: self.layer1(nodes), adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks,input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout,workers=workers,pool=self.pool)
        self.layer3 = SageLayer(id2feat=lambda nodes: self.layer2(nodes), adj_list=adj, dis_list=dis,
                                restart_prob=restart_prob, num_walks=num_walks,input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout,workers=workers,pool=self.pool)
        '''

        self.layer2 = SageLayer(id2feat=lambda nodes: self.id2node[nodes],
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=input_dim,
                                output_dim=embed_dim, device=device, dropout=dropout, id=2, adj_queues=adj_queues,
                                dis_queues=dis_queues, all_adj_queues=all_adj_queues, all_dis_queues=all_dis_queues)
        self.layer1 = SageLayer(id2feat=lambda nodes: self.layer2(nodes),
                                restart_prob=restart_prob, num_walks=num_walks, input_dim=embed_dim,
                                output_dim=embed_dim, device=device, dropout=dropout, id=1, adj_queues=adj_queues,
                                dis_queues=dis_queues, all_adj_queues=all_adj_queues, all_dis_queues=all_dis_queues)

    def forward(self, nodes):
        feats = self.layer1(nodes)
        return feats

    def setup(self, X, adj, dis):
        self.id2node = X
        self.layer1.set_adj(adj, dis)
        self.layer2.set_adj(adj, dis)
