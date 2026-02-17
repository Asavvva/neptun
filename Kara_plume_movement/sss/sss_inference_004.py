from netCDF4 import Dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from datetime import date, timedelta
from tqdm import tqdm
from glob import glob
import pickle
import random
import os
import fnmatch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Tuple, List, Type, Dict, Any

from SGDR import CosineAnnealingWarmRestarts
from mish import Mish
from MyResidualNetwork import MyResNet, MyBasicBlock
from MyDataPreparationMix_norm import CustomDataset
from autoencoder import Encoder, Decoder


class InferenceDataset(TorchDataset):
    def __init__(self, dates_dict1, path1, dates_dict2, path2):
        self.dates_and_sources = []
        self._add_dates(dates_dict1, path1)
        self._add_dates(dates_dict2, path2)
    
    def _add_dates(self, dates_dict, source_path):
        for year_str in sorted(dates_dict.keys()):
            year = int(year_str)
            bounds = dates_dict[year_str]

            m_start = int(bounds['start'][:2])
            d_start = int(bounds['start'][2:])
            m_finish = int(bounds['finish'][:2])
            d_finish = int(bounds['finish'][2:])

            start_date = date(year, m_start, d_start)
            finish_date = date(year, m_finish, d_finish)

            current = start_date
            while current <= finish_date:
                date_str = current.strftime('%Y%m%d')
                self.dates_and_sources.append((date_str, source_path))
                current += timedelta(days=1)

    def make_data(self, idx):
        date_str, path = self.dates_and_sources[idx]
        file = f'{path}/{date_str[:4]}/{date_str}.pkl'
        with open(file, 'rb') as f:
            _, _, sss = pickle.load(f)
        sss = np.where(np.isnan(sss), -999, sss)
        sss = sss[np.newaxis, :, :]
        mask = np.where(sss == -999, 0, 1)
        return sss, mask, date_str, path

    def __len__(self):
        return len(self.dates_and_sources)
    
    def __getitem__(self, idx):
        sss, mask, date_str, path = self.make_data(idx)
        return sss, mask, date_str, path
    

def make_out_path(in_path, date_str, out_path='/app/Kara_plume_movement/extracted_features/extracted_sss_004', suffix="_encoded"):
    base = os.path.basename(in_path)
    out_name = f'{date_str}{suffix}'
    out_path_base = f'{out_path}/{base}/'

    return out_path_base, out_name


if __name__ == '__main__':
    dates1 = {
        '2010': {'start': '0711', 'finish': '1009'},
        '2011': {'start': '0625', 'finish': '1029'},
        '2012': {'start': '0701', 'finish': '1105'},
        '2013': {'start': '0713', 'finish': '1018'},
        '2014': {'start': '0716', 'finish': '1023'},
        '2015': {'start': '0630', 'finish': '1025'},
        '2016': {'start': '0709', 'finish': '1031'},
        '2017': {'start': '0715', 'finish': '1020'},
        '2018': {'start': '0801', 'finish': '1031'},
        '2019': {'start': '0710', 'finish': '1025'},
        '2020': {'start': '0707', 'finish': '1031'},
        '2021': {'start': '0710', 'finish': '1025'},
        '2022': {'start': '0701', 'finish': '1031'},
        '2023': {'start': '0720', 'finish': '1010'},
    }

    dates2 = {
        '2015': {'start': '0630', 'finish': '1025'},
        '2016': {'start': '0709', 'finish': '1031'},
        '2017': {'start': '0715', 'finish': '1020'},
        '2018': {'start': '0801', 'finish': '1031'},
        '2019': {'start': '0710', 'finish': '1025'},
        '2020': {'start': '0707', 'finish': '1031'},
        '2021': {'start': '0710', 'finish': '1025'},
        '2022': {'start': '0701', 'finish': '1031'},
        '2023': {'start': '0720', 'finish': '1010'},
    }

    batch_size = 16
    dataset = InferenceDataset(dates_dict1=dates1, path1='/mnt/hippocamp/asavin/data/ESACCI/ESACCI_norm',
                            dates_dict2=dates2, path2='/mnt/hippocamp/asavin/data/SSS_ESACCI_grid/SSS_ESACCI_grid_norm_free_ice_dates')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    run_name = 'sss_pre_autoencoder_run004'
    device = torch.device('cuda:1')
    encoder = torch.load(f'/app/Kara_plume_movement/sss/models/model_{run_name}_encoder.pth', map_location=torch.device('cpu'));
    encoder.eval();
    encoder = encoder.cuda()

    with torch.no_grad():
        for batch_data in dataloader:
            data, mask, date_str, path = batch_data
            data_gpu = data.to(device='cuda', dtype=torch.float)            
            encoded_data = encoder.forward(data_gpu)
            encoded_data_cpu = encoded_data.detach().cpu().numpy()

            for i, (date_str, path) in enumerate(zip(date_str, path)):
                out_path_base, out_name = make_out_path(path, date_str, out_path='/app/Kara_plume_movement/extracted_features/extracted_sss_004', suffix="_encoded")
                os.makedirs(f'{out_path_base}/{date_str[:4]}', exist_ok=True)
                with open(f'{out_path_base}/{date_str[:4]}/{out_name}', "wb") as f:
                    pickle.dump(encoded_data[i], f)
