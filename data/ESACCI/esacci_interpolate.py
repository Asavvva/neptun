import numpy as np
from scipy.interpolate import griddata
from netCDF4 import Dataset
import os, fnmatch
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


def interpolate(lat, lon, sss_):
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    data_flat = sss_.ravel()

    points = np.column_stack((lon_flat, lat_flat))

    dlat = 0.125
    dlon = 0.5

    lat_new = np.arange(70, 80 + dlat, dlat)
    lon_new = np.arange(55, 105 + dlon, dlon)

    lon_small2d, lat_small2d = np.meshgrid(lon_new, lat_new)

    # Линейная интерполяция (по сути билинейная в треугольниках триангуляции)
    data_small2d = griddata(
        points,
        data_flat,
        (lon_small2d, lat_small2d),
        method='linear'
    )

    return data_small2d


def make_data(f):
    data = Dataset(f, 'r')
    
    latitude = np.asarray(data['lat'])
    longitude = np.asarray(data['lon'])
    sss = np.asarray(data['sss'])[0]

    lon, lat, sss_ = select_data(longitude=longitude, latitude=latitude, sss=sss, borders=[83,67,30,110])
    sss_grid = interpolate(lat, lon, sss_)

    return sss_grid


def select_and_save(year, start, finish):
    files = find_files(f'/mnt/hippocamp/DATA/sattelite/ESACCI/v05.5/NHv5.5/7days/{year}/', '*.nc')
    files.sort()

    files = [f for f in files if f[-13:-9] >= start and f[-13:-9] <= finish]
    
    for file in files:
        sss_grid = make_data(file)
        
        try:
            with open(f'/app/data/ESACCI/ESACCI_interpolated/{year}/{file[-17:-9]}.pkl', 'wb') as file:
                pickle.dump(sss_grid, file)

        except:
            os.makedirs(f'/app/data/ESACCI/ESACCI_interpolated/{year}', exist_ok=True)
            with open(f'/app/data/ESACCI/ESACCI_interpolated/{year}/{file[-17:-9]}.pkl', 'wb') as file:
                pickle.dump(sss_grid, file)


select_and_save(year = 2011, start='0601', finish='1130')
