from torch.autograd import Variable
import time
import copy
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
print(sys.path)

from util.plan_to_tree import Node, parse_dep_tree_text
from util.prase_tree2node_leaf import (
    treeInterpolation,
    hierarchical_embeddings,
    upward_ca,
    tree2NodeLeafmat,
)
from model.encoder import attention, WeightedAggregation, LayerNorm, Reshape, clones


class DecoderLinear(nn.Module):
    def __init__(self, d_feature, d_model):
        super(DecoderLinear, self).__init__()
        self.query_linear = nn.Linear(d_model, d_feature)
        self.key_linear = nn.Linear(d_model, d_feature)
        self.vlaue_linear = nn.Linear(d_model, d_feature)

    def forward(self, x, target):
        value = self.value_linear(x)
        key = self.key_linear(x)
        query = self.query_linear(target)
        return value, key, query


class DecoderAttentionScaledDot(nn.Module):
    def __init__(self, d_feature, d_model, dropout=0.1):
        super(DecoderAttentionScaledDot, self).__init__()
        # self.decoderLiner = DecoderLinear(d_feature, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q_target, node_k, leaf_k, mask=None):
        Aqn = attention(query=q_target, key=node_k, mask=mask, dropout=self.dropout)
        Aql = attention(query=q_target, key=leaf_k, mask=mask, dropout=self.dropout)
        return Aqn, Aql


class DecoderAttention(nn.Module):
    def __init__(self, d_feature, d_model):
        super(DecoderAttention, self).__init__()
        self.linear = DecoderLinear(d_feature=d_feature, d_model=d_model)
        self.scaledDot = DecoderAttentionScaledDot(d_feature=d_feature, d_model=d_model)
        self.weightedAgg = WeightedAggregation(d_feature)

    def forward(self, root, node, leaf, target):
        node_v, node_k, node_q = self.linear(node, target)
        leaf_v, leaf_k, leaf_q = self.linear(leaf, target)

        # node_q == leaf_q is target

        Aqn, Aql = self.scaledDot(node_q, node_k, leaf_k)

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

        # !!!! dim
        Attq = F.softmax(
            torch.matmul(
                torch.cat(Aqn, Aql), torch.cat(node_hat.double(), leaf_hat, dim=-2)
            )
        )
        return Attq


class DecoderLayer(nn.Module):
    def __init__(self, d_feature, d_model, d_ff):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_feature)
        self.norm2 = LayerNorm(d_feature)
        self.decoderAttention = DecoderAttention(d_feature, d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, root, node_x, leaf_x, target):
        # !!! target + mask(norm(attention(target)))
        x = self.decoderAttention(root, node_x, leaf_x, target)
        x = x + self.norm1(x)
        x = self.feed_forward(x)
        x = x + self.norm2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_feature, d_model, d_ff, N):
        super(Decoder, self).__init__()
        self.reshape = Reshape(d_feature=d_feature, d_model=d_model)
        self.layers = clones(DecoderLayer, N)

    def forward(self, root, node_x, leaf_x, target):
        target = self.reshape(target)
        for layer in self.layers:
            target = layer(root, node_x, leaf_x, target)
        return target
