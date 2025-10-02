import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import datetime
from mpl_toolkits.basemap import Basemap
import os


def drawing(longitude, latitude, adt, time, i):
    fig = plt.figure(figsize=(12, 12), dpi = 300)
    m = Basemap(width=1200000, height=1500000,
                resolution='l', projection='aea',
                lat_1=60, lat_2=65, lon_0=76, lat_0=77)

    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='white')
    m.drawparallels(np.arange(-80.,90.,10.), labels=[False,True,True,False])
    m.drawmeridians(np.arange(-180.,180.,20.), labels=[True,True,False,True])
    m.drawmapboundary(fill_color='white')

    X, Y = np.meshgrid(longitude, latitude)
    m.scatter(X, Y, c=adt[i,:,:],
              cmap='jet', latlon=True)
    cbar = plt.colorbar(label='ADT', orientation='vertical', shrink=0.30)
    plt.title(f'{time[i].date()}')
    font = {'size'   : 12}
    plt.rc('font', **font)
    ax = plt.gca()
    try:
        plt.savefig(f'/mnt/hippocamp/asavin/Plume_description/adt/pictures/{time[i].year}/{time[i]}.png')
    except:
        os.makedirs(f'/mnt/hippocamp/asavin/Plume_description/adt/pictures/{time[i].year}')
        plt.savefig(f'/mnt/hippocamp/asavin/Plume_description/adt/pictures/{time[i].year}/{time[i]}.png')
    plt.show()
    plt.clf()
    plt.close('all')


data = Dataset('/mnt/hippocamp/asavin/data/ssh/adt_1993-2022_daily.nc', 'r')

start_time = datetime.datetime(1970, 1, 1)
time = np.asarray(data['time'])
time = np.asarray([start_time + datetime.timedelta(seconds=int(t)) for t in time])

longitude = np.asarray(data['longitude'])
latitude = np.asarray(data['latitude'])
adt = np.asarray(data['adt'])

indices_ = np.where([date.year == 2015 for date in time])[0]
time_ = time[indices_]
adt_ = adt[indices_]

for i in range(243, 244):
    drawing(longitude, latitude, adt_, time_, i=i)
