# -*- coding: utf-8 -*-

import argparse
import os
from train import train
import torch
import weight_method
import utils
from data_loader import Dataset
from torch_geometric.data import DataLoader
import generate_decoy



def prepair_training_args(args_dict):
    parser = argparse.ArgumentParser(description='Description of your program')
    for arg, kwargs in args_dict.items():
        parser.add_argument(arg, **kwargs)
    args = parser.parse_args()
    if args.dataset == 'ransomware':
        args.num_user_features = 2
    elif args.dataset == 'book-crossing':
        args.num_user_features = 3

    return args



def prepair_dataloader(args):

    sep = ','
    dataset = Dataset('../data/', args.dataset, args.rating_file, sep, args)

    data_num = dataset.data_N()  # 交互数量
    feature_num = dataset.feature_N()  # 属性特征数量
    train_index, val_index = dataset.stat_info['train_test_split_index']  # 训练数据、验证数据的分界索引

    # 分割交互图
    train_loader = DataLoader(dataset[:train_index], batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(dataset[train_index:val_index], batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(dataset[val_index:], batch_size=args.batch_size, num_workers=0)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    datainfo = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'all_data': dataset_loader,
        'feature_num': feature_num,
        'data_num': [len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)]
    }

    return datainfo



def get_train_vector(model):

    feature_dict = utils.load_dict('..\\data\\ransomware\\attr_dict.pkl')
    user_dict = utils.load_dict('..\\data\\ransomware\\user_dict.pkl')
    item_dict = utils.load_dict('..\\data\\ransomware\\item_dict.pkl')

    all_user_index_in_feature_dict_list = [value for key, value in feature_dict.items() if 'ransomware_name' in key]
    user_in_train = []
    for index in model.node_dict.keys():
        if int(index) in all_user_index_in_feature_dict_list:
            user_in_train.append(int(index))

    all_file_index_in_feature_dict_list = [value for key, value in feature_dict.items() if 'file_name_..' in key]
    file_in_train = []
    for index in model.node_dict.keys():
        if int(index) in all_file_index_in_feature_dict_list:
            file_in_train.append(int(index))

    train_user_list = []
    for user in user_in_train:
        train_user = []
        for key, value in user_dict.items():
            if user == value['name']:
                train_user.append(user)
                attr = value['attribute']
                train_user.extend(attr)
        train_user_list.append(train_user)

    train_file_list = []
    for file in file_in_train:
        train_file = []
        for key, value in item_dict.items():
            if file == value['title']:
                train_file.append(file)
                attr = value['attribute']
                train_file.extend(attr)
        train_file_list.append(train_file)

    i = 0
    train_user_vec_list = []
    train_user_vec_dict = {}
    while i < len(train_user_list):
        sub_train_user_list = train_user_list[i]
        sub_train_user_vec_list = []
        for user_index in sub_train_user_list:
            vec = model.node_dict[f'{user_index}']
            sub_train_user_vec_list.append(vec)
        train_user_vec_list.append(sub_train_user_vec_list)
        train_user_vec_dict[f'{user_in_train[i]}'] = sub_train_user_vec_list
        i += 1

    j = 0
    train_file_vec_list = []
    train_file_vec_dict = {}
    while j < len(train_file_list):
        sub_train_file_list = train_file_list[j]
        sub_train_file_vec_list = []
        for file_index in sub_train_file_list:
            vec = model.node_dict[f'{file_index}']
            sub_train_file_vec_list.append(vec)
        train_file_vec_list.append(sub_train_file_vec_list)
        train_file_vec_dict[f'{file_in_train[j]}'] = sub_train_file_vec_list
        j += 1

    return train_user_vec_dict, train_file_vec_dict, model.user_dict



def get_hit_rate(decoy_file_name_list, mode):
    # Using evaluation metrics to test the effectiveness of bait files
    # Select the mode you want to test,
    # for example, 'test' means comparing with interaction records from the real test set:

    if mode == 'test':
        pkl = 'ransomware_file_reaction_dict_for_test_dict.pkl'
    ransomware_file_dict = utils.load_dict(f'../data/{pkl}')

    result_dict = {}

    j = 0  # num of ransomware samples for testing
    for ransomware, attacked_file_list in ransomware_file_dict.items():
        hit_dict = {}
        sub_dict = {}
        attacked_file_list = list(attacked_file_list['file'])
        for decoy_file in decoy_file_name_list:
            i = 0
            for file in attacked_file_list:
                if decoy_file in file:
                    hit_dict[f'{file}'] = i + 1
                    break
                else:
                    i += 1
        if len(hit_dict) != 0:
            min_file_key = min(hit_dict, key=hit_dict.get)
            min_value = hit_dict[min_file_key]
            sub_dict['hit file'] = min_file_key
            sub_dict['loss file count'] = min_value - 1
            result_dict[f'{ransomware}'] = sub_dict
        else:
            sub_dict['hit file'] = ''
            sub_dict['loss file count'] = 100
            result_dict[f'{ransomware}'] = []
        j += 1

    hit_count_list = []  # average file loss list
    miss_list = []  # undetected ransomware list
    for key, value in result_dict.items():
        try:
            loss_file_count = value['loss file count']
            hit_count_list.append(loss_file_count)
        except Exception as e:
            miss_list.append(key)

    average_loss_count = sum(hit_count_list) / len(hit_count_list)

    return result_dict, j, average_loss_count, miss_list



if __name__ == '__main__':

    # Select the mode you want to test,
    # for example, 'test' means comparing with interaction records from the real test set
    mode = 'test'

    args_dict = {
        '--dataset': {'type': str, 'default': 'ransomware', 'help': 'which dataset to use'},
        '--rating_file': {'type': str, 'default': f'interaction_ratings.csv', 'help': 'reaction record file'},
        '--dim': {'type': int, 'default': 64, 'help': 'dimension of entity and relation embeddings'},
        '--l2_weight': {'type': float, 'default': 1e-5, 'help': 'weight of the l2 regularization term'},
        '--lr': {'type': float, 'default': 0.001, 'help': 'learning rate'},
        '--batch_size': {'type': int, 'default': 128, 'help': 'batch size'},
        '--n_epoch': {'type': int, 'default': 50, 'help': 'the number of epochs'},
        '--hidden_layer': {'type': int, 'default': 256, 'help': 'neural hidden layer'},
        '--num_user_features': {'type': int, 'default': 2, 'help': 'the number of user attributes'},
        '--random_seed': {'type': int, 'default': 2024, 'help': 'size of common item be counted'}
    }
    args = prepair_training_args(args_dict)

    datainfo = prepair_dataloader(args)

    # using GNN to learn the interaction between ransomware and user files in the training set
    model_path = '..\\checkpoint\\trained_model.pkl'
    if not os.path.exists(model_path):
        train(args, datainfo, model_path)
    main_model = torch.load(model_path)
    utils.print_formatted_current_time(f'have loaded the trained model form {model_path}')

    # using a linear model to learn the weights of each attribute
    train_user_vec_dict, train_file_vec_dict, user_vec_dict = get_train_vector(main_model)
    model_path = '..\\checkpoint\\trained_weight_model.pkl'
    if not os.path.exists(model_path):
        weight_method.train_weight(user_vec_dict, train_user_vec_dict, train_file_vec_dict, model_path)
    weight_model = torch.load(model_path)
    utils.print_formatted_current_time(f'have loaded trained weight model from {model_path}')

    # generating decoy files
    decoy_file_name_list = generate_decoy.generate_decoy_files(main_model, weight_model, user_vec_dict)
    utils.print_formatted_current_time(f' {len(decoy_file_name_list)} decoy files have been selected')

    # Using evaluation metrics to test the effectiveness of bait files
    result_dict, j, average_loss_count, miss_list  = get_hit_rate(decoy_file_name_list, mode)

    # Presenting downstream tasks of test results ... ...
