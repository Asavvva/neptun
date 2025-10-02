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
    '''
    Класс обрабатывает данные /app/data/adt/adt_1993-2024_daily_n80_s70_w55_e105.nc
    Данные представлены на регулярной сетке, шаг 0.125 (примерно)
    Данные есть даже в присутствие льда, информация про лед есть в маске flag_ice
    Выдаются двумерные данные adt и геострофических скоростей, маска суши и маска льда
    Маска суши: 1 - море, 0 - суша; маска льда: 1 - чистая вода, 0 - лед или суша
    Чтобы получить 0 там, где данных нет, нужно поэлементно умножить данные на маску льда
    '''

    def __init__(self, data):
        adt = np.array(data.variables['adt'])
        ugos = np.array(data.variables['ugos'])
        vgos = np.array(data.variables['vgos'])
        flag_ice = np.asarray(data.variables['flag_ice'])
        
        self.adt = adt
        self.ugos = ugos
        self.vgos = vgos
        self.flag_ice = flag_ice
    
    def __len__(self):
        return self.adt.shape[0]
    
    def get_data(self, index):
        adt = self.adt[index, :]
        ugos = self.ugos[index, :]
        vgos = self.vgos[index, :]
        
        data2D = np.stack([adt, ugos, vgos])
        land_mask = np.where(np.isnan(data2D), 0, 1)
        data2D[land_mask==0] = 0

        ice_mask = np.where((np.isnan(data2D)) | (self.flag_ice == 1), 0, 1)
        
        return data2D, land_mask, ice_mask
    
    def __getitem__(self, index):
        data2D, mask, ice_mask = self.get_data(index)
        
        return [data2D, mask, ice_mask]


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
