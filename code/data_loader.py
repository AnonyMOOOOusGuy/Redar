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

        # init some parameters here
        # ... ...


    def process(self):

        # process triain dataset and construct graphs here
        # ... ...


