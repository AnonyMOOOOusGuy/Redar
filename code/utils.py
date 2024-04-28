# -*- coding: utf-8 -*-
import csv
import pickle
from datetime import datetime
import numpy as np


def print_formatted_current_time(string):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]
    print(f"[{formatted_time}] {string}")



def write_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f'{key}: {value}\n')



def save_dict_as_pickle(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)



def load_dict(file_path):
    with open(file_path, 'rb') as f:
        object = pickle.load(f)

    return object



def construct_interaction_matrix(train_user_vec_dict, train_file_vec_dict):
    # extract training labels
    user_dict = load_dict('..\\data\\ransomware\\user_dict.pkl')
    item_dict = load_dict('..\\data\\ransomware\\item_dict.pkl')

    user_index_in_feature_dict = train_user_vec_dict.keys()
    user_index_in_user_dict = []
    for user_index in user_index_in_feature_dict:
        for key, value in user_dict.items():
            if value['name'] == int(user_index):
                user_index_in_user_dict.append(key)
                break

    file_index_in_feature_dict = train_file_vec_dict.keys()
    file_index_in_item_dict = []
    for file_index in file_index_in_feature_dict:
        for key, value in item_dict.items():
            if value['title'] == int(file_index):
                file_index_in_item_dict.append(key)
                break

    tuple_list = load_csv_ratings_2_tuple_list(f'..\\data\\ransomware\\interaction_ratings.csv')

    num_users = len(user_index_in_user_dict)
    num_files = len(file_index_in_item_dict)

    interaction_matrix = np.zeros((num_users, num_files), dtype=int)
    for user, file, interaction in tuple_list:
        if str(user) in user_index_in_user_dict and str(file) in file_index_in_item_dict:
            user_index = user_index_in_user_dict.index(str(user))
            file_index = file_index_in_item_dict.index(str(file))
            interaction_matrix[user_index][file_index] = interaction

    return interaction_matrix

def load_csv_ratings_2_tuple_list(csv_path):
    tuple_list = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tuple_list.append(tuple(int(item) for item in row))

    return tuple_list


def find_decoy_files_by_attr(picked_attr):
    feature_dict = load_dict('../data/ransomware/attr_dict.pkl')
    item_dict = load_dict('../data/ransomware/item_dict.pkl')
    decoy_file_index_list = []
    for attr in picked_attr:
        for feature_key, feature_value in feature_dict.items():
            if attr in feature_key:
                attr_index_in_feature_dict = feature_value
                for item_key, item_value in item_dict.items():
                    attr_list = list(item_value['attribute'])
                    if attr_index_in_feature_dict in attr_list:
                        file_index_in_feature_dict = item_value['title']
                        decoy_file_index_list.append(file_index_in_feature_dict)

    return decoy_file_index_list


