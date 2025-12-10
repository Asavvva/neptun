import torch
from torch.utils.data import Dataset as TorchDataset
from datetime import date, timedelta
import numpy as np
import pickle


class CustomDataset(TorchDataset):
    """
    Dataset, который принимает словарь вида:
    {
        '2010': {'start': '0711', 'finish': '1009'},
        '2011': {'start': '0625', 'finish': '1029'},
        ...
    }
    и возвращает строки дат в формате 'YYYYMMDD' для всех дней
    между start и finish (включительно) для каждого года.
    """

    def __init__(self, dates_dict):
        self.dates_dict = dates_dict
        self._make_dates()

    def _make_dates(self):
        dates = []
        for year_str in sorted(self.dates_dict.keys()):
            year = int(year_str)
            bounds = self.dates_dict[year_str]

            # парсим MMDD -> month, day
            m_start = int(bounds['start'][:2])
            d_start = int(bounds['start'][2:])
            m_finish = int(bounds['finish'][:2])
            d_finish = int(bounds['finish'][2:])

            start_date = date(year, m_start, d_start)
            finish_date = date(year, m_finish, d_finish)

            current = start_date
            while current <= finish_date:
                # сохраняем в формате 'YYYYMMDD'
                dates.append(current.strftime('%Y%m%d'))
                current += timedelta(days=1)

        self.dates = dates

    def make_data(self, idx):
        file = (f'/mnt/hippocamp/asavin/data/ESACCI/ESACCI_norm/{self.dates[idx][:4]}/{self.dates[idx]}.pkl')
        with open(file, 'rb') as f:
            _, _, sss = pickle.load(f)
        
        sss = np.where(np.isnan(sss), -999, sss)
        sss = sss[np.newaxis, :, :]
        mask = np.where(sss == -999, 0, 1)

        return sss, mask
    
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):
        sss, mask = self.make_data(idx)

        return sss, mask
