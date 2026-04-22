from netCDF4 import Dataset
import datetime
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import fnmatch
import re
from collections import OrderedDict
from tqdm import tqdm


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


R  = 6371

def distance(lat1, lon1, lat2, lon2):
    lat_av = np.deg2rad(lat1 - lat2)/2
    lon_av = np.deg2rad(lon1 - lon2)/2
    dd = np.sin(lat_av)**2 + np.cos(np.deg2rad(lat2)) * np.cos(np.deg2rad(lat1)) * np.sin(lon_av)**2
    dd = 2 * R * np.arcsin(np.clip(np.sqrt(dd), -1.0, 1.0))
    return dd


################################ Общие настройки ################################


year = 2015
radius_km = 10
radius_t = 3
deltaT = 6


################################ Найти имена файлов SMAP ################################


insitu = pd.read_csv(f'/mnt/hippocamp/asavin/data/sss_insitu/Data_insitu_{year}.csv')
files = find_files(f'/mnt/hippocamp/DATA/satellite/SMAP_V6.0/L2C/{year}/', '*.nc')
files.sort()

insitu['Time'] = pd.to_datetime(insitu['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

file_info = []

for fname in files:
    m = re.search(r'(\d{8}T\d{6})', fname)
    if m:
        file_time = pd.to_datetime(m.group(1), format='%Y%m%dT%H%M%S')
        file_info.append((fname, file_time))

def find_files_for_time(t):
    if pd.isna(t):
        return ''
    
    matched = [
        fname for fname, ftime in file_info
        if abs(ftime - t) <= pd.Timedelta(hours=deltaT)
    ]
    
    return '; '.join(matched)

insitu['matched_files'] = insitu['Time'].apply(find_files_for_time)


################################ Открывать данные SMAP ################################


def get_SMAP_data(filename):
    SMAP_data = Dataset(f'{filename}', 'r')

    celtime = np.asarray(SMAP_data['time'])
    cellat = np.asarray(SMAP_data['cellat'])
    cellon = np.asarray(SMAP_data['cellon'])
    time = np.array([datetime.datetime(2000, 1 , 1) + datetime.timedelta(seconds=t) for t in celtime.ravel()]).reshape(celtime.shape)

    gland = np.asarray(SMAP_data['gland'])
    fland = np.asarray(SMAP_data['fland'])
    gice_est = np.asarray(SMAP_data['gice_est'])
    surtep = np.asarray(SMAP_data['surtep'])
    winspd = np.asarray(SMAP_data['winspd'])
    windir = np.asarray(SMAP_data['windir'])
    solar_flux = np.asarray(SMAP_data['solar_flux'])
    sunglt = np.asarray(SMAP_data['sunglt'])
    monglt = np.asarray(SMAP_data['monglt'])
    tb_sur0_sic = np.asarray(SMAP_data['tb_sur0_sic'])
    sss_smap = np.asarray(SMAP_data['sss_smap'])
    iqc_flag = np.asarray(SMAP_data['iqc_flag'])
    tb_consistency = np.asarray(SMAP_data['tb_consistency'])

    windir_cos = np.cos(windir)
    windir_sin = np.sin(windir)

    ### добавляем "look" для климатических переменных 
    gice_est = np.repeat(gice_est[:, :, np.newaxis], 2, axis=2)
    surtep = np.repeat(surtep[:, :, np.newaxis], 2, axis=2)
    winspd = np.repeat(winspd[:, :, np.newaxis], 2, axis=2)
    windir_cos = np.repeat(windir_cos[:, :, np.newaxis], 2, axis=2)
    windir_sin = np.repeat(windir_sin[:, :, np.newaxis], 2, axis=2)
    solar_flux = np.repeat(solar_flux[:, :, np.newaxis], 2, axis=2)
    
    ### раскрываем brightness temperature в 4 поляризациях
    tb_sur0_sic_0 = tb_sur0_sic[:,:,:,0]
    tb_sur0_sic_1 = tb_sur0_sic[:,:,:,1]
    tb_sur0_sic_2 = tb_sur0_sic[:,:,:,2]
    tb_sur0_sic_3 = tb_sur0_sic[:,:,:,3]

    vars = {
        'time': time, 'cellat': cellat, 'cellon': cellon,
        'gland': gland, 'fland': fland, 'gice_est_2l': gice_est,
        'surtep_2l': surtep, 'winspd_2l': winspd, 'windir_cos_2l': windir_cos, 'windir_sin_2l': windir_sin,
        'solar_flux_2l': solar_flux, 'sunglt': sunglt, 'monglt': monglt,
        'tb_sur0_sic_0': tb_sur0_sic_0, 'tb_sur0_sic_1': tb_sur0_sic_1, 'tb_sur0_sic_2': tb_sur0_sic_2, 'tb_sur0_sic_3': tb_sur0_sic_3,
        'sss_smap': sss_smap, 'iqc_flag': iqc_flag, 'tb_consistency': tb_consistency
    }

    return vars


################################ Хранение открытых файлов в кеше, чтобы не открывать их много раз для соседних строк ################################


file_cache = OrderedDict()
max_cache_size = 10

def get_file_from_cache(fname):
    global file_cache
    if fname in file_cache:
        # переносим ключ в конец, чтобы считать его недавним
        file_cache.move_to_end(fname)
        return file_cache[fname]
    else:
        data = get_SMAP_data(fname)
        file_cache[fname] = data
        # если кеш стал слишком большим - удаляем старый элемент
        if len(file_cache) > max_cache_size:
            file_cache.popitem(last=False)
        return data


################################ Основной цикл ################################


results = []

for idx, row in tqdm(insitu.iterrows(), total=insitu.shape[0]):
    dt_ref = pd.to_datetime(row['Time'])
    lat_ref, lon_ref = row['latitude'], row['longitude']
    sss_ref = row['Salinity']
    uuid_ref = row['UUID']
    file_list = row['matched_files'].split(';')

    for fname in file_list:
        if not fname.strip():
            continue
        
        nc_data = get_file_from_cache(fname)

        # Получаем плоские массивы для всех переменных сразу
        flat_vars = {var: nc_data[var].ravel() for var in nc_data.keys()}

        # Считаем расстояния до всех точек
        dist = distance(lat_ref, lon_ref, flat_vars['cellat'], flat_vars['cellon'])
        flat_times = pd.to_datetime(flat_vars['time'])
        time_delta = np.abs((flat_times - dt_ref).total_seconds())

        # Маска времени: только те, что меньше 10 часов
        mask_time = time_delta <= 10 * 3600

        # Основная маска: расстояние + радиус времени (radius_t)
        radius_t_sec = pd.Timedelta(hours=radius_t).total_seconds()
        mask = (dist < radius_km) & (time_delta < radius_t_sec) & mask_time

        idxs = np.where(mask)[0]

        for i in idxs:
            result = {
                'Time': dt_ref,
                'latitude': lat_ref,
                'longitude': lon_ref,
                'Salinity': sss_ref,
                'UUID': uuid_ref,
                'fname': fname,
            }
            # Добавляем все переменные из списка
            for var in nc_data.keys():
                result[var] = flat_vars[var][i]
            results.append(result)

# Сбор в DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(f'/mnt/hippocamp/asavin/data/sss_insitu_SMAP_V6.0/sss_insitu_SMAP_{year}_{radius_km}km_{radius_t}hrs.csv', index=False)
