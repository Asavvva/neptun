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


with open(f'/mnt/hippocamp/asavin/data/wind/wind_products_data/wind_products_norm_params.pkl', 'rb') as file:
    wind_products_norm_params = pickle.load(file)

wind_files = find_files('/mnt/hippocamp/asavin/data/wind/wind_products_arrays_kara_n80_s70_w55_e105', '*.pkl')
wind_files.sort()

for file in wind_files:
    with open(file, 'rb') as f:
        wind_data = pickle.load(f)

    wind_data = np.transpose(wind_data, (1, 0, 2, 3))
    u10, v10, r, r2, ru, rv = wind_data

    u10 = (u10 - wind_products_norm_params['u10_mean']) / wind_products_norm_params['u10_std']
    v10 = (v10 - wind_products_norm_params['v10_mean']) / wind_products_norm_params['v10_std']

    tr = np.log1p(r/wind_products_norm_params['r_mean'])
    zr = (tr - wind_products_norm_params['tr_mean']) / wind_products_norm_params['tr_std']
    r = zr

    tr2 = np.log1p(r2/wind_products_norm_params['r2_5p'])
    zr2 = (tr2 - wind_products_norm_params['tr2_mean']) / wind_products_norm_params['tr2_std']
    r2 = zr2

    tru = np.arcsinh(ru/wind_products_norm_params['ru_median'])
    zru = (tru - wind_products_norm_params['tru_mean']) / wind_products_norm_params['tru_std']
    ru = zru

    trv = np.arcsinh(rv/wind_products_norm_params['rv_median'])
    zrv = (trv - wind_products_norm_params['trv_mean']) / wind_products_norm_params['trv_std']
    rv = zrv

    wind = np.stack([u10, v10, r, r2, tru, trv])
    wind = np.transpose(wind, (1, 0, 2, 3))

    with open(f'/mnt/hippocamp/asavin/data/wind/wind_products_arrays_kara_norm_n80_s70_w55_e105/{file[-11:]}', 'wb') as file:
        pickle.dump(wind, file)
