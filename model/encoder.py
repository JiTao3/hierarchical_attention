import copy
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
# print(sys.path)

from util.plan_to_tree import Node, parse_dep_tree_text
from util.prase_tree2node_leaf import treeInterpolation, upward_ca, tree2NodeLeafmat
from util.dataset import PlanDataset


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(feature), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, mask=None, dropout=None):
    """get score"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


class TreeAttentionLinear(nn.Module):
    def __init__(self, d_feature, d_model, dropout=0.1):
        super(TreeAttentionLinear, self).__init__()
        self.query_linear = nn.Linear(d_feature, d_model)
        self.key_linear = nn.Linear(d_feature, d_model)
        self.vlaue_linear = nn.Linear(d_feature, d_model)

    def forward(self, x):
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.vlaue_linear(x)
        return q, k, v


class TreeAttentionScaledDot(nn.Module):
    def __init__(self, d_feature, dropout=0.1):
        super(TreeAttentionScaledDot, self).__init__()
        # !!! use different dropout ???
        self.dropout = nn.Dropout(p=dropout)
        # self.leafLinear = nn.Linear(d_feature, d_feature)

    def forward(self, node_q, node_k, leaf_q, leaf_k, mask=None):
        Anl = attention(query=node_q, key=leaf_k, mask=mask, dropout=self.dropout)
        Ann = attention(query=node_q, key=node_k, mask=mask, dropout=self.dropout)
        All = attention(query=leaf_q, key=leaf_k, mask=mask, dropout=self.dropout)
        Aln = attention(query=leaf_q, key=node_k, mask=mask, dropout=self.dropout)

        return Anl, Ann, All, Aln


class WeightedAggregation(nn.Module):
    def __init__(self, d_feature):
        super(WeightedAggregation, self).__init__()
        # !!!
        self.u_s = nn.Parameter(torch.ones(d_feature, requires_grad=True))
        self.register_parameter("U_s", self.u_s)
        self.d_featuer = d_feature

    def forward(self, leaf, upward_ca_vec):
        # omega size leaf * d
        omega = torch.matmul(leaf, self.u_s)
        # upward_ca_vec size node * leaf * d
        omega_shape = omega.shape[-1]
        weighted_aggregation_vec = upward_ca_vec * omega.reshape([1, omega_shape, 1])
        # no_zero shape node * 1
        # weight_aggregation_vec shape is node*leaf*d
        weighted_aggregation_vec = torch.sum(weighted_aggregation_vec, dim=1)
        # weight_aggregation_vec shape is node*d

        nozero_div = (np.count_nonzero(upward_ca_vec.detach().numpy(), axis=(1, 2)) + 1e-6) / self.d_featuer
        no_zero = 1 / nozero_div
        # no_zero_shape =
        no_zero = torch.from_numpy(no_zero)

        weighted_aggregation_vec = weighted_aggregation_vec * torch.unsqueeze(no_zero, 1)

        return weighted_aggregation_vec


class TreeAttention(nn.Module):
    def __init__(self, d_feature, d_model):
        super(TreeAttention, self).__init__()
        self.linear = TreeAttentionLinear(d_feature=d_feature, d_model=d_model)
        self.scaledDot = TreeAttentionScaledDot(d_feature=d_feature)
        self.weightAgg = WeightedAggregation(d_feature=d_feature)

    def forward(self, root: Node, node, leaf):

        node_q, node_k, node_v = self.linear(node)
        leaf_q, leaf_k, leaf_v = self.linear(leaf)
        Anl, Ann, All, Aln = self.scaledDot(node_q, node_k, leaf_q, leaf_k)
        # !!!! node_hat = ???

        # but you should keep the order of node?!!!
        # the order of node_q & node and leaf_q & leaf should be same
        # you should use parse tree 2 node leaf plan_tree_leaves_node to keep the order

        interpolation_vec = treeInterpolation(root=root, leaf=leaf_v, node=node_v)

        # node + 1 * leaf * d
        # you should use parse tree 2 node leaf plan_tree_leaves_node to keep the order

        upward_ca_vec = upward_ca(interpolation_vec)
        # upward_ca_tensor = torch.from_numpy(upward_ca_vec)

        node_hat = self.weightAgg(leaf, upward_ca_vec)
        leaf_hat = leaf_v

        # 1)!!! node_hat = ???

        # 2) cat the matrix and return attn and attl
        # !!! DIM
        # !!! mask
        # AnnAnl = torch.cat((Ann, Anl),dim=-1)
        # leafnodehat = torch.cat((node_hat.float(), leaf_hat),dim=-2)
        Attn = torch.matmul(
            F.softmax(torch.cat((Ann, Anl), dim=-1), dim=-2),
            torch.cat((node_hat, leaf_hat), dim=-2),
        )
        Attl = torch.matmul(
            F.softmax(torch.cat((Aln, All), dim=-1), dim=-2),
            torch.cat((node_hat, leaf_hat), dim=-2),
        )
        return Attn, Attl


class Reshape(nn.Module):
    def __init__(self, d_feature, d_model):
        super(Reshape, self).__init__()
        self.reshape = nn.Linear(d_feature, d_model)

    def forward(self, x):
        return self.reshape(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_feature, d_model, d_ff):
        super(EncoderLayer, self).__init__()
        # self.reshape = nn.Linear(d_feature, d_model)
        self.treeattn = TreeAttention(d_feature, d_model)
        # Wo
        # !!! d
        self.linear = nn.Linear(d_model, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, root, node, leaf):
        Attn, Attl = self.treeattn(root, node, leaf)
        Attno, Attlo = self.linear(Attn), self.linear(Attl)
        node_x = node + self.norm1(Attno)
        leaf_x = leaf + self.norm2(Attlo)
        feed_node_x = self.feed_forward(node_x)
        feed_leaf_x = self.feed_forward(leaf_x)
        node_x = node_x + self.norm2(feed_node_x)
        leaf_x = leaf_x + self.norm2(feed_leaf_x)
        return node_x, leaf_x


class Encoder(nn.Module):
    def __init__(self, d_feature, d_model, d_ff, N):
        super(Encoder, self).__init__()
        self.reshape = Reshape(d_feature=d_feature, d_model=d_model)
        self.layers = clones(
            EncoderLayer(d_feature=d_model, d_model=d_model, d_ff=d_ff), N=N
        )
        self.forward_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, root, node, leaf):
        node = self.reshape(node)
        leaf = self.reshape(leaf)

        for layer in self.layers:
            node, leaf = layer(root, node, leaf)

        x = torch.cat((node, leaf), dim=-2)
        # max pool
        x = torch.max(x, dim=-2, keepdim=True)[0]
        x = self.forward_net(x)
        return x.squeeze(-1)


if __name__ == "__main__":
    encoder = Encoder(d_feature=9 + 6 + 64, d_model=512, d_ff=512, N=2).double()
    dataset = PlanDataset(root_dir="data/deep_plan")

    tree, nodemat, leafmat, label = dataset[0]
    print(nodemat.shape, leafmat.shape)

    x = encoder(tree, nodemat.double(), leafmat.double())
    print(x)
