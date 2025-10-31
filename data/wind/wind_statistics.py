from netCDF4 import Dataset
import pickle
import numpy as np


def mean_wind(month, lat1, lat2, lon1, lon2):
    file = f'/mnt/hippocamp/DATA/ERA5/w10/era5_uv10m_{month}.nc'
    data = Dataset(file, 'r')

    u10 = data.variables['u10'][:,lat1:lat2,lon1:lon2]
    v10 = data.variables['v10'][:,lat1:lat2,lon1:lon2]

    wind_speed_hourly_2d = np.asarray([np.sqrt(a**2 + b**2) for (a,b) in zip(u10, v10)])
    az_to_hourly_2d = (np.degrees(np.arctan2(u10, v10)) + 360) % 360
    az_hourly_2d = (az_to_hourly_2d + 180) % 360

    u10_monthly = u10.mean()
    v10_monthly = v10.mean()
    wind_speed_monthly = np.sqrt(u10_monthly**2 + v10_monthly**2)
    az_to_monthly = (np.degrees(np.arctan2(u10_monthly, u10_monthly)) + 360) % 360
    az_monthly = (az_to_monthly + 180) % 360
    data.close()

    return {'wind_speed_hourly_2d': wind_speed_hourly_2d,
            'az_to_hourly_2d': az_to_hourly_2d,
            'az_hourly_2d': az_hourly_2d,
            'wind_speed_monthly': wind_speed_monthly,
            'az_to_monthly': az_to_monthly,
            'az_monthly': az_monthly
            }


lon_min_era5, lon_max_era5, lat_min_era5, lat_max_era5 = 65, 85, 73, 78
lat1, lat2 = (90-lat_max_era5)*4, (90-lat_min_era5)*4+1
lon1, lon2 = lon_min_era5*4, lon_max_era5*4+1

months = [
    "2000-07", "2000-08", "2000-09", "2000-10",
    "2001-07", "2001-08", "2001-09", "2001-10",
    "2002-07", "2002-08", "2002-09", "2002-10",
    "2003-07", "2003-08", "2003-09", "2003-10",
    "2004-07", "2004-08", "2004-09", "2004-10",
    "2005-07", "2005-08", "2005-09", "2005-10",
    "2006-07", "2006-08", "2006-09", "2006-10",
    "2007-07", "2007-08", "2007-09", "2007-10",
    "2008-07", "2008-08", "2008-09", "2008-10",
    "2009-07", "2009-08", "2009-09", "2009-10",
    "2010-07", "2010-08", "2010-09", "2010-10",
    "2011-07", "2011-08", "2011-09", "2011-10",
    "2012-07", "2012-08", "2012-09", "2012-10",
    "2013-07", "2013-08", "2013-09", "2013-10",
    "2014-07", "2014-08", "2014-09", "2014-10",
    "2015-07", "2015-08", "2015-09", "2015-10",
    "2016-07", "2016-08", "2016-09", "2016-10",
    "2017-07", "2017-08", "2017-09", "2017-10",
    "2018-07", "2018-08", "2018-09", "2018-10",
    "2019-07", "2019-08", "2019-09", "2019-10",
    "2020-07", "2020-08", "2020-09", "2020-10",
    "2021-07", "2021-08", "2021-09", "2021-10",
    "2022-07", "2022-08", "2022-09", "2022-10",
    "2023-07", "2023-08", "2023-09", "2023-10",
    "2024-07", "2024-08", "2024-09", "2024-10"
]

for month in months:
    wind_statistics = mean_wind(month, lat1, lat2, lon1, lon2)
    with open(f'/mnt/hippocamp/asavin/data/wind/wind_statistics_kara_n78_s73_w65_e85/{month}.pkl', 'wb') as file:
        pickle.dump(wind_statistics, file)
