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

        self.make_norm_data()
    
    def __len__(self):
        return self.adt.shape[0]
    
    def nan_mean_var(self, x: np.ndarray, ddof: int = 0):
        """
        Возвращает (mean, var) по элементам x, игнорируя NaN.
        ddof=0 -> population variance, ddof=1 -> sample variance.
        Если все значения NaN, возвращает (np.nan, np.nan).
        """

        mean = np.nanmean(x)
        var = np.nanvar(x, ddof=ddof)

        return float(mean), float(var)
    
    def nan_normalize(self, x: np.ndarray, mean: float | None = None, std: float | None = None, ddof: int = 0, eps: float = 1e-12):
        """
        Нормализует (z-score) только не-NaN элементы.
        Возвращает (x_norm, mean, std).
        """

        x = np.asarray(x)
        if mean is None or std is None:
            m, v = self.nan_mean_var(x, ddof=ddof)
            mean = m
            std = np.sqrt(max(v, 0.0)) if np.isfinite(v) else 0.0

        x_norm = x.copy()
        valid = ~np.isnan(x)
        if valid.any():
            x_norm[valid] = (x[valid] - mean) / (std + eps)

        return x_norm, float(mean), float(std)
    
    def nan_denormalize(self, x_norm: np.ndarray, mean: float, std: float, eps: float = 1e-12):
        """
        Обратное преобразование: денормализует только не-NaN элементы.
        """

        x_norm = np.asarray(x_norm)
        x = x_norm.copy()
        valid = ~np.isnan(x_norm)
        if valid.any():
            x[valid] = x_norm[valid] * (std + eps) + mean

        return x
    
    def make_norm_data(self):
        self.adt, _, _ = self.nan_normalize(self.adt)
        self.ugos, _, _ = self.nan_normalize(self.ugos)
        self.vgos, _, _ = self.nan_normalize(self.vgos)
    
    def get_data(self, index):
        adt = self.adt[index, :]
        ugos = self.ugos[index, :]
        vgos = self.vgos[index, :]
        flag_ice = self.flag_ice[index, :]

        data2D = np.stack([adt, ugos, vgos])
        land_mask = np.where(np.isnan(data2D), 0, 1)
        data2D[land_mask == 0] = 0

        ice_mask = np.where(np.isnan(adt) | (flag_ice == 1), 0, 1)
        ice_mask = np.repeat(np.expand_dims(ice_mask, axis=0), data2D.shape[0], axis=0)
        
        return data2D, land_mask, ice_mask
    
    def __getitem__(self, index):
        data2D, mask, ice_mask = self.get_data(index)
        
        return data2D, mask, ice_mask


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
