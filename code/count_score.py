# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
import utils


def get_file_attr_vec(node_dict):
    # Obtaining Filename Attribute Vector Representation,
    # take file attribute size_rank for example

    attr_dict = utils.load_dict('..\\data\\ransomware\\attr_dict.pkl')
    file_size_rank_index = []
    # other file attributes ... ...

    for key, value in attr_dict.items():
        if 'file_size_rank_' in key:
            file_size_rank_index.append(value)
        # other file attributes
        # ... ...
        continue

    file_size_rank_vec_list = []
    # other file attribute vector lists
    # ... ...
    for file_size_rank in file_size_rank_index:
        file_size_rank_vec_list.append(node_dict[f'{file_size_rank}'])
    # handling other file attributes
    # ... ...

    return file_size_rank_index, file_size_rank_vec_list, \
        # return other file attibute index lists and vector lists
    # ... ...



def count_similar_score(user_vec_list, file_attr_vec_list):
    # Calculate the score for each ransomware for different
    # values of the current file attributes

    user_vec = np.array(user_vec_list)
    file_attr_vec_list = np.array(file_attr_vec_list)
    user_attr_similarity = cosine_similarity(user_vec, file_attr_vec_list)
    return user_attr_similarity


def count_picked_score(matrix, scale_factor):
    # Multiplying weights by scores to obtain selection scores
    # for different values of each attribute

    matrix = torch.tensor(matrix)
    scaled_matrix = matrix * scale_factor
    column_sums = torch.sum(scaled_matrix, dim=0)
    column_means = column_sums / scaled_matrix.shape[0]
    column_means_list = column_means.detach().numpy()
    return column_means_list


def pick_attr(picked_score_list, attr_list):
    attr_dict = utils.load_dict('../data/ransomware/attr_dict.pkl')
    sorted_indices = np.argsort(picked_score_list)
    top_n_indices = sorted_indices[-n:]  # choose n top score attributes as candidate attributes of decoy files
    picked_attr = []
    for i in top_n_indices:
        attr_index_in_feature_dict = attr_list[i]
        for key, value in attr_dict.items():
            if value == attr_index_in_feature_dict:
                picked_attr.append(key)

    return picked_attr


def find_certain_attr(model, user_vec_dict, attr_weight_list):
    # Forming a collection of candidate bait file properties

    # other file attibute index lists and vector lists ... ..., \
    file_size_rank_index, file_size_rank_vec_list = get_file_attr_vec(model.node_dict)

    vecs = [file_size_rank_vec_list,
            # other file attribute vector list ... ...
            ]
    user_vec_list = list(user_vec_dict.values())
    similar_scores = [count_similar_score(user_vec_list, vec) for vec in vecs]

    # other file attribute similar score \
    # ... ...
    user_size_rank_similar = similar_scores

    score_lists = []
    for similar, weight in zip(
            [user_size_rank_similar,
             # other file attribute similar score
             # ... ...
             ], attr_weight_list):
        score_lists.append(count_picked_score(similar, weight))

    # other file attribute similar score
    # ... ...
    user_size_rank_picked_score_list = score_lists

    picked_size_rank_attr = pick_attr(user_size_rank_picked_score_list, file_size_rank_index)
    # other picked file attribute lists
    # ... ...

    return picked_size_rank_attr
            # return other selected file attributes
            # ... ...
