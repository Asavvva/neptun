from netCDF4 import Dataset
import numpy as np
import datetime
from abc import abstractmethod

import pickle
import random
import pandas as pd

import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, List, Type, Dict, Any


class CustomDataset(TorchDataset):
    def __init__(self,
                 wind_files_pkl,
                 n_files):
        self.wind_files_pkl = wind_files_pkl
        self.n_files = n_files
        
    def _get_months(self):
        self.current_months = random.sample(self.wind_files_pkl, self.n_files)
        
    def _load_file(self):
        wind_list = []
        for file in self.current_months:
            with open(f'{file}', 'rb') as f:
                wind = pickle.load(f)
                wind_list.append(wind)
        
        self.wind_array = np.concatenate(wind_list)
        
    def make_new_data(self):
        self._get_months()
        self._load_file()
    
    def __len__(self):
        return self.wind_array.shape[0]
    
    def clear_cache(self):
        self.current_months = None
        self.wind_array = None
    
    def get_data(self, index):
        data2D = self.wind_array[index, :]
        
        return data2D
        
    def __getitem__(self, index):
        data2D = self.get_data(index)
        
        return data2D


class Sampler:
    def __init__(self, index, shuffle=False):
        self.index = index
        self.shuffle = shuffle

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        ids = np.arange(len(self.index))
        if self.shuffle:
            np.random.shuffle(ids)
        for i in ids:
            yield self.index[i]
