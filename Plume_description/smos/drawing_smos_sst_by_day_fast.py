from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import fnmatch
import datetime


def drawing(longitude, latitude, sst, time):
    fig = plt.figure(figsize=(12, 12), dpi = 300)
    m = Basemap(width=1200000, height=1500000,
                resolution='l', projection='aea',
                lat_1=60, lat_2=65, lon_0=76, lat_0=77)

    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='white')
    m.drawparallels(np.arange(-80.,90.,10.), labels=[False,True,True,False])
    m.drawmeridians(np.arange(-180.,180.,20.), labels=[True,True,False,True])
    m.drawmapboundary(fill_color='white')

    m.scatter(longitude, latitude, c=sst,
              cmap='jet', latlon=True, vmax=12)
    cbar = plt.colorbar(label='SST', orientation='vertical', shrink=0.30)
    plt.title(f'{time.date()}')
    font = {'size'   : 12}
    plt.rc('font', **font)
    ax = plt.gca()
    try:
        plt.savefig(f'/mnt/hippocamp/asavin/Plume_description/smos/pictures/pictures_SST/{time.year}/{time.date()}.png')
    except:
        os.makedirs(f'/mnt/hippocamp/asavin/Plume_description/smos/pictures/pictures_SST/{time.year}')
        plt.savefig(f'/mnt/hippocamp/asavin/Plume_description/smos/pictures/pictures_SST/{time.year}/{time.date()}.png')
    plt.show()
    plt.clf()
    plt.close('all')


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


dd = []
for root, dirs, files in os.walk('/mnt/hippocamp/DATA/sattelite/SMOS/L2OS/MIR_OSUDP2_nc/2011/09'):
    dd.append(root)

dd.sort()

for date in dd[1:]:
    files = find_files(date, '*.nc')
    if len(files) > 0:
        time = datetime.datetime.strptime(date[-11:].strip("/"), "%Y/%m/%d")
        
        lons = np.array([])
        lats = np.array([])
        SSTs = np.array([])
        for file in files:
            try:
                data = Dataset(file, 'r')

                lat = np.asarray(data['Latitude'])
                lon = np.asarray(data['Longitude'])
                SST = np.asarray(data['SST'])
                mask = (SST > -999) & (lon >= 50) & (lon <= 110) & (lat >= 67) & (lat <= 85)

                lons = np.concatenate((lons, lon[mask]))
                lats = np.concatenate((lats, lat[mask]))
                SSTs = np.concatenate((SSTs, SST[mask]))

                coords = np.vstack((lats, lons)).T
                unique_coords, indices, inverse_indices = np.unique(coords, axis=0, return_index=True, return_inverse=True)
                sst_sum = np.bincount(inverse_indices, weights=SSTs)
                sst_count = np.bincount(inverse_indices)
                sst_mean = sst_sum / sst_count

            except:
                pass

        drawing(unique_coords[:, 1], unique_coords[:, 0], sst_mean, time)
