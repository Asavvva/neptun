import torch
from torch.utils.data import Dataset as TorchDataset
from datetime import date, timedelta
import numpy as np
from netCDF4 import Dataset


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

    def __init__(self, dates_dict, borders):
        self.borders = borders
        dates = []

        for year_str in sorted(dates_dict.keys()):
            year = int(year_str)
            bounds = dates_dict[year_str]

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

    def select_data(self, longitude, latitude, sss):
        n, s, w, e, = self.borders
        mask = (latitude >= s) & (latitude <= n) & (longitude >= w) & (longitude <= e)
        
        rows_with_data = np.any(mask, axis=1)
        cols_with_data = np.any(mask, axis=0)

        row_idx = np.where(rows_with_data)[0]
        col_idx = np.where(cols_with_data)[0]

        lon = longitude[row_idx.min():row_idx.max() + 1,
                        col_idx.min():col_idx.max() + 1]
        
        lat = latitude[row_idx.min():row_idx.max() + 1,
                col_idx.min():col_idx.max() + 1]
        
        sss_ = sss[row_idx.min():row_idx.max() + 1,
                col_idx.min():col_idx.max() + 1]
        
        return lon, lat, sss_

    def make_data(self, idx):
        n, s, w, e, = self.borders
        file = (f'/mnt/hippocamp/DATA/sattelite/ESACCI/v05.5/NHv5.5/7days/{self.dates[idx][:4]}/' +
                f'ESACCI-SEASURFACESALINITY-L4-SSS-POLAR-MERGED_OI_7DAY_RUNNINGMEAN_DAILY_25kmEASE2_NH-{self.dates[idx]}-fv5.5.nc')
        data = Dataset(file, 'r')
        
        latitude = np.asarray(data['lat'])
        longitude = np.asarray(data['lon'])
        sss = np.asarray(data['sss'])[0]

        data.close()
        
        lon, lat, sss_ = self.select_data(longitude=longitude, latitude=latitude, sss=sss)
        sss_ = np.where((lon < w) | (lon > e) | (lat > n) | (lat < s), np.nan, sss_)
        sss_ = np.where(np.isnan(sss_), 0, sss_)
        sss_ = sss_[np.newaxis, :, :]
        mask = np.where(sss_ == 0, 0, 1)

        return sss_, mask

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        sss, mask = self.make_data(idx)

        return sss, mask
