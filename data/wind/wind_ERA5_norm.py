from netCDF4 import Dataset
import pickle
import os
import fnmatch
import numpy as np


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


def norm_wind(x, _mean, _std):
    norm_x = (x - _mean) / _std
    return norm_x


def load_file(filename, lon_min_era5 = 55, lon_max_era5 = 105, lat_min_era5 = 70, lat_max_era5 = 80):
    lat_min_era5, lat_max_era5 = (90-lat_max_era5)*4, (90-lat_min_era5)*4+1
    lon_min_era5, lon_max_era5 = lon_min_era5*4, lon_max_era5*4+1

    data = Dataset(filename, 'r')
    u10 = np.array(data.variables['u10'][:,
                                         lat_min_era5:lat_max_era5,
                                         lon_min_era5:lon_max_era5])
    v10 = np.array(data.variables['v10'][:,
                                         lat_min_era5:lat_max_era5,
                                         lon_min_era5:lon_max_era5])
    u10 = norm_wind(u10, norm_params['mean_u10'], norm_params['std_u10'])
    v10 = norm_wind(v10, norm_params['mean_v10'], norm_params['std_v10'])
    wind = np.stack([u10, v10])
    wind = np.transpose(wind, (1, 0, 2, 3))
    data.close()
    
    return wind


with open('/mnt/hippocamp/asavin/data/wind/normalizer_params_wind_kara', 'rb') as f:
    norm_params = pickle.load(f)

wind_files = find_files('/mnt/hippocamp/DATA/ERA5/w10', '*.nc')
wind_files.sort()
wind_files = [file for file in wind_files if (file[-5:-3] <= '11' and file[-5:-3] >= '06' and
                                              file[-10:-6] >= '1979' and file[-10:-6] < '2023')]

for file in wind_files:
    wind = load_file(file)
    with open(f'/mnt/hippocamp/asavin/data/wind/wind_arrays_kara_norm_n80_s70_w55_e105/{file[-10:-3]}.pkl', 'wb') as file:
        pickle.dump(wind, file)
