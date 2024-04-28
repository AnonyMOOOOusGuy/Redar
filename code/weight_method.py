# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import utils
import numpy as np
import torch.optim as optim

""" define a linear model to learn the weights of different attr classes """
class SingleLayerNetwork(nn.Module):
    def __init__(self, input_size, output_size):

        # define proper model of weight training model
        # ... ...

    def forward(self, X, user_vec_dict):

        # weight learning process
        # ... ...


def train_weight(user_vec_dict, train_user_vec_dict, train_file_vec_dict, model_path):
    """
    Redar weight training model
    """

    # # learning process,
    # including:
    # receive learnt embeddings, define loss function, bp process, evaluate prediction results and save model
    # ... ...
    # ... ...


def calculate_accuracy(predicted_matrix, true_interaction_matrix, threshold=0.5):

    # calculate the accuracy of the linear model
    # ... ...