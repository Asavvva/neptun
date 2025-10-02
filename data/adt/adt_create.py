import os
import fnmatch
import xarray as xr


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


files = find_files('/mnt/hippocamp/asavin/data/adt/data/', '*.nc')
files.sort()

datasets = [xr.open_dataset(fp) for fp in files]
combined_dataset = xr.concat(datasets, dim='time')
combined_dataset.to_netcdf('/mnt/hippocamp/asavin/data/adt/adt_1993-2024_daily_n80_s70_w55_e105.nc')
