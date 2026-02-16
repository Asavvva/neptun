import torch
from torch.utils.data import Dataset as TorchDataset
from datetime import date, timedelta
import numpy as np
import pickle


class CustomDataset(TorchDataset):
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
        return sss, mask

    def __len__(self):
        return len(self.dates_and_sources)
    
    def __getitem__(self, idx):
        sss, mask = self.make_data(idx)
        return sss, mask
