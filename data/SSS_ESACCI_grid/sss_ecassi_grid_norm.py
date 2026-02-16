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
    data_norm = (data_sinh - (-2.4215977)) / 0.061461017 # mean = -2.4215977; std = 0.061461017

    return data_norm

def make_data(f, borders, norm=False):
    n, s, w, e, = borders
    data = Dataset(f, 'r')
    
    latitude = np.asarray(data['lat'])
    longitude = np.asarray(data['lon'])
    sss = np.asarray(data['sss'])

    data.close()
    
    lon, lat, sss_ = select_data(longitude=longitude, latitude=latitude, sss=sss, borders=borders)
    sss_ = np.where((lon < w) | (lon > e) | (lat > n) | (lat < s), np.nan, sss_)

    if norm:
        sss_ = norm_data(sss_)

    return lon, lat, sss_


def select_and_save(year, borders, norm=False):
    files = find_files('/mnt/hippocamp/asavin/data/SSS_ESACCI_grid/SSS_ESACCI_grid_data', '*.nc')
    files.sort()
    
    for file in files:
        day = file[-6:-3]
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day) -1)
        sss_grid = make_data(file, borders, norm=norm)
        
        try:
            with open(f'/app/data/SSS_ESACCI_grid/SSS_ESACCI_grid_norm/{year}/{date.strftime("%Y%m%d")}.pkl', 'wb') as file:
                pickle.dump(sss_grid, file)

        except:
            os.makedirs(f'/app/data/SSS_ESACCI_grid/SSS_ESACCI_grid_norm/{year}', exist_ok=True)
            with open(f'/app/data/SSS_ESACCI_grid/SSS_ESACCI_grid_norm/{year}/{date.strftime("%Y%m%d")}.pkl', 'wb') as file:
                pickle.dump(sss_grid, file)


years = range(2015, 2024)

for year in years:
    array = select_and_save(year=year, borders=[80,70,55,105], norm=True)
    print(f'{year} complited')
