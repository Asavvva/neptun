import numpy as np
from scipy.interpolate import griddata
from netCDF4 import Dataset
import os, fnmatch
import datetime
import pickle


def find_files(directory, pattern, maxdepth=None):
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\\\', os.sep)
                if maxdepth is None:
                    flist.append(filename)
                else:
                    if filename.count(os.sep)-directory.count(os.sep) <= maxdepth:
                        flist.append(filename)
    return flist


def select_data(longitude, latitude, sss, borders):
    n, s, w, e, = borders
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


def norm_data(data):
    sss = data
    data_upd = np.where(sss < 0, np.nan, sss)
    inv_data = np.where(data_upd != 0, 1 / data_upd, np.nan)
    data_log = np.log(inv_data)
    data_sinh = np.arcsinh(data_log/0.6)
    data_norm = (data_sinh - (-2.4005454)) / 0.09315018 # mean = -2.4005454; std = 0.09315018

    return data_norm

def make_data(f, borders, norm=False):
    n, s, w, e, = borders
    data = Dataset(f, 'r')
    
    latitude = np.asarray(data['lat'])
    longitude = np.asarray(data['lon'])
    sss = np.asarray(data['sss'])[0]

    data.close()
    
    lon, lat, sss_ = select_data(longitude=longitude, latitude=latitude, sss=sss, borders=borders)
    sss_ = np.where((lon < w) | (lon > e) | (lat > n) | (lat < s), np.nan, sss_)

    if norm:
        sss_ = norm_data(sss_)

    return lon, lat, sss_


def select_and_save(year, start, finish, borders, norm=False):
    files = find_files(f'/mnt/hippocamp/DATA/sattelite/ESACCI/v05.5/NHv5.5/7days/{year}/', '*.nc')
    files.sort()

    files = [f for f in files if f[-13:-9] >= start and f[-13:-9] <= finish]
    
    for file in files:
        sss_grid = make_data(file, borders, norm=norm)
        
        try:
            with open(f'/app/data/ESACCI/ESACCI_norm/{year}/{file[-17:-9]}.pkl', 'wb') as file:
                pickle.dump(sss_grid, file)

        except:
            os.makedirs(f'/app/data/ESACCI/ESACCI_norm/{year}', exist_ok=True)
            with open(f'/app/data/ESACCI/ESACCI_norm/{year}/{file[-17:-9]}.pkl', 'wb') as file:
                pickle.dump(sss_grid, file)


dates = {
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


all_dates = [] 

for year_str, bounds in dates.items():
    year = int(year_str)
    # парсим MMDD -> month, day
    m_start = int(bounds['start'][:2])
    d_start = int(bounds['start'][2:])
    m_finish = int(bounds['finish'][:2])
    d_finish = int(bounds['finish'][2:])
    
    start_date = datetime.date(year, m_start, d_start)
    finish_date = datetime.date(year, m_finish, d_finish)

    current = start_date
    while current <= finish_date:
        # формат YYYYMMDD без разделителей
        all_dates.append(current.strftime('%Y%m%d'))
        current += datetime.timedelta(days=1)


for key in dates.keys():
    array = select_and_save(year=int(key), start=dates[key]['start'], finish=dates[key]['finish'], borders=[80,70,55,105], norm=True)
    print(f'{key} complited')
