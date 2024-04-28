# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
import utils


def get_file_attr_vec(node_dict):

    # Obtaining Filename Attribute Vector Representation
    # ... ...


def count_similar_score(user_vec_list, file_attr_vec_list):

    # Calculate the score for each ransomware for different values of the current file attributes
    # ... ...


def count_picked_score(matrix, scale_factor):

    # multiply weights by scores to obtain selection scores for different values of each attribute
    # ... ...


def pick_attr(picked_score_list, attr_list):

    # choose n top score attributes as candidate attributes of decoy files
    # ... ...


def find_certain_attr(model, user_vec_dict, attr_weight_list):

    # Forming a collection of candidate decoy file properties
    # ... ...
