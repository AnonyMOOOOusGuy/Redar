# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
import numpy as np

""" Define Internal Interaction Network """
class inner_GNN(MessagePassing):
    def __init__(self, dim, hidden_layer):

        # define proper model of inner GNN
        # ... ...

    def forward(self, x, edge_index, edge_weight=None):

        # inner GNN learning process
        # ... ...

    def message(self, x_i, x_j, edge_weight):

        # message passing process
        # ... ...


    def update(self, aggr_out):
        # update learnt embeddings with new messages
        # ... ...

""" Define Cross-Interaction Network """
class cross_GNN(MessagePassing):

    def __init__(self, dim, hidden_layer):

        # define proper model of outer GNN
        # ... ...

    def forward(self, x, edge_index, edge_weight=None):

        # outer GNN learning process
        # ... ...

    def message(self, x_i, x_j, edge_weight):

        # message passing process
        # ... ...

    def update(self, aggr_out):

        # update learnt embeddings with new messages
        # ... ...


class Redar(nn.Module):
    """
    Redar main model
    """
    def __init__(self, args, n_features, device):

        # define model parameters here
        # ... ...

    def forward(self, data, is_training=True):

        # get training data , process and predict
        # ... ...
        # ... ...

    def split_batch(self, batch, user_node_num):

        # split training batch data
        # ... ...

    def outer_offset(self, batch, user_node_num, outer_edge_index):

        # Align the learned cross-interaction embeddings
        # ... ...