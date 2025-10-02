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
    def __init__(self, data, non_nan_threshold_value):
        adt = np.array(data.variables['adt'])
        ugos = np.array(data.variables['ugos'])
        vgos = np.array(data.variables['vgos'])
        
        non_nan_counts = np.sum(~np.isnan(adt), axis=(1, 2))
        adt_upd = adt[non_nan_counts >= non_nan_threshold_value]
        ugos_upd = ugos[non_nan_counts >= non_nan_threshold_value]
        vgos_upd = vgos[non_nan_counts >= non_nan_threshold_value]
        
        self.adt_upd = adt_upd
        self.ugos_upd = ugos_upd
        self.vgos_upd = vgos_upd
    
    def __len__(self):
        return self.adt_upd.shape[0]
    
    def get_data(self, index):
        adt = self.adt_upd[index, :]
        ugos = self.ugos_upd[index, :]
        vgos = self.vgos_upd[index, :]
        
        data2D = np.stack([adt, ugos, vgos])
        mask = np.where(np.isnan(data2D), 0, 1)
        data2D[mask==0] = 0
        
        return data2D, mask
        
    def __getitem__(self, index):
        data2D, mask = self.get_data(index)
        
        return [data2D, mask]


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
