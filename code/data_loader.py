# -*- coding: utf-8 -*-
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os



class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, rating_file, sep, args, transform=None, pre_transform=None):

        self.path = root
        self.dataset = dataset
        self.rating_file = rating_file
        self.sep = sep
        self.store_backup = True
        self.args = args
        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.stat_info = torch.load(self.processed_paths[1])
        self.data_num = self.stat_info['data_num']
        self.attr_num = self.stat_info['attr_num']

    @property
    def raw_file_names(self):
        return ['{}{}/user_dict.pkl'.format(self.path, self.dataset),
                '{}{}/item_dict.pkl'.format(self.path, self.dataset),
                '{}{}/attr_dict.pkl'.format(self.path, self.dataset),
                '{}{}/{}'.format(self.path, self.dataset, self.rating_file)]

    @property
    def processed_file_names(self):
        return ['{}/{}.dataset'.format(self.dataset, self.dataset),
                '{}/{}.statinfo'.format(self.dataset, self.dataset)]

    def download(self):
        pass

    def attr_N(self):
        return self.attr_num

    def data_N(self):
        return self.data_num


    def process(self):
        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.attrfile = self.raw_file_names[2]
        self.ratingfile  = self.raw_file_names[3]
        graphs, stat_info = self.read_data()

        if not os.path.exists(f"{self.path}processed/{self.dataset}"):
            os.mkdir(f"{self.path}processed/{self.dataset}")
            print(f"data has saved to {self.path}processed/{self.dataset}")
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(stat_info, self.processed_paths[1])


    def read_data(self):
        self.user_dict = pickle.load(open(self.userfile, 'rb'))
        self.item_dict = pickle.load(open(self.itemfile, 'rb'))
        self.user_key_type = type(list(self.user_dict.keys())[0])
        self.item_key_type = type(list(self.item_dict.keys())[0])
        attr_dict = pickle.load(open(self.attrfile, 'rb'))
        ratings_df = pd.read_csv(self.ratingfile, sep=self.sep, header=None)
        train_df, test_df = train_test_split(ratings_df, test_size=0.4, random_state=self.args.random_seed)
        test_df, valid_df = train_test_split(test_df, test_size=0.5, random_state=self.args.random_seed)

        if self.store_backup:
            backup_path = f"{self.path}{self.dataset}/split_data_backup/"
            if not os.path.exists(backup_path):
                os.mkdir(backup_path)
            train_df.to_csv(f'{backup_path}train_data.csv', index=False)
            valid_df.to_csv(f'{backup_path}valid_data.csv', index=False)
            test_df.to_csv(f'{backup_path}test_data.csv', index=False)

        train_graphs = self.process_dataset(train_df, 'train')
        valid_graphs = self.process_dataset(valid_df, 'valid')
        test_graphs = self.process_dataset(test_df, 'test')
        graphs = train_graphs + valid_graphs + test_graphs

        stat_info = {}
        stat_info['data_num'] = len(graphs)
        stat_info['attr_num'] = len(attr_dict)
        stat_info['train_test_split_index'] = [len(train_graphs), len(train_graphs) + len(valid_graphs)]
        return graphs, stat_info


    def process_dataset(self, df, dataset_name):
        graphs = self.data_2_graphs(df, dataset=dataset_name)
        return graphs


    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs = []
        error_num = 0
        processed_graphs = 0
        num_graphs = ratings_df.shape[0]
        one_per = int(num_graphs/1000)
        percent = 0.0
        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print(f"Processing [{dataset}]: {percent/10.0}%, {processed_graphs}/{num_graphs}", end="\r")
                percent += 1
            processed_graphs += 1
            line = ratings_df.iloc[i]
            user_index = self.user_key_type(line[0])
            item_index = self.item_key_type(line[1])
            rating = int(line[2])
            if item_index not in self.item_dict or user_index not in self.user_dict:
                error_num += 1
                continue

            user_id = self.user_dict[user_index]['name']
            item_id = self.item_dict[item_index]['title']
            user_attr_list = self.user_dict[user_index]['attribute']
            item_attr_list = self.item_dict[item_index]['attribute']
            user_list = [user_id] + user_attr_list
            item_list = [item_id] + item_attr_list

            graph = self.construct_graphs(user_list, item_list, rating)
            graphs.append(graph)
        return graphs

    def construct_graphs(self, user_list, item_list, rating):
        u_n = len(user_list)
        i_n = len(item_list)
        inner_edge_index = [[], []]
        for i in range(u_n + i_n):
            for j in range(i, u_n + i_n):
                inner_edge_index[0].append(i)
                inner_edge_index[1].append(j)
        outer_edge_index = [[], []]
        for i in range(u_n):
            for j in range(i_n):
                outer_edge_index[0].append(i)
                outer_edge_index[1].append(u_n + j)
        inner_edge_index = torch.LongTensor(inner_edge_index)
        inner_edge_index = to_undirected(inner_edge_index)
        outer_edge_index = torch.LongTensor(outer_edge_index)
        outer_edge_index = to_undirected(outer_edge_index)
        graph = self.construct_graph(user_list + item_list, inner_edge_index, outer_edge_index, rating)

        return graph


    def construct_graph(self, node_list, edge_index_inner, edge_index_outer, rating):
        x = torch.LongTensor(node_list).unsqueeze(1)
        rating = torch.FloatTensor([rating])
        return Data(x=x, edge_index=edge_index_inner, edge_attr=torch.transpose(edge_index_outer, 0, 1), y=rating)