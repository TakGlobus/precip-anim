import xarray as xr
import zarr
import numpy as np
from scipy.interpolate import griddata

def update(ds):
    # update latitude
    ds = ds.sortby("lat", ascending=False)
    # update longitude
    lons = ds.lon.values
    lons[lons < 0] += 360
    ds = ds.assign_coords(lon=lons)
    ds = ds.sortby("lon", ascending=True)
    return ds

# Paths to the files
zarr_file = "imerg-2020-july-1d-3600x1800.zarr"
era5_file = "era5-2020-july-1d-1440x721.nc"

# Open the Zarr file
imerg_ds = xr.open_zarr(zarr_file)
imerg_ds = update(imerg_ds)

# Open the ERA5 NetCDF file
era5_ds = xr.open_dataset(era5_file)

# Extract latitude and longitude from both datasets
imerg_lats = imerg_ds['lat'].values
imerg_lons = imerg_ds['lon'].values
era5_lats = era5_ds['latitude'].values
era5_lons = era5_ds['longitude'].values

# Create a meshgrid for ERA5 latitude and longitude
imerg_ds_interp = imerg_ds.interp(lat=era5_lats, lon=era5_lons)

imerg_ds_interp.to_netcdf("imerg-2020-july-1d-1440x721.nc", mode="w")

print(imerg_ds_interp)
"""_summary_
<xarray.Dataset> Size: 129MB
Dimensions:        (time: 31, lon: 1440, lat: 721)
Coordinates:
  * time           (time) datetime64[ns] 248B 2020-07-01 ... 2020-07-31
  * lat            (lat) float64 6kB 90.0 89.75 89.5 ... -89.5 -89.75 -90.0
  * lon            (lon) float64 12kB 0.0 0.25 0.5 0.75 ... 359.2 359.5 359.8
Data variables:
    precipitation  (time, lon, lat) float32 129MB dask.array<chunksize=(1, 1440, 91), meta=np.ndarray>
Attributes:
    BeginDate:       2020-07-01
    BeginTime:       00:00:00.000Z
    EndDate:         2020-07-01
    EndTime:         23:59:59.999Z
    FileHeader:      StartGranuleDateTime=2020-07-01T00:00:00.000Z;\nStopGran...
    InputPointer:    3B-HHR.MS.MRG.3IMERG.20200701-S000000-E002959.0000.V07B....
    title:           GPM IMERG Final Precipitation L3 1 day 0.1 degree x 0.1 ...
    DOI:             10.5067/GPM/IMERGDF/DAY/07
    ProductionTime:  2024-01-03T02:35:41.620Z
    history:         2025-03-31 16:54:53 GMT hyrax-1.17.1 https://gpm1.gesdis...
    history_json:    [{"$schema":"https:\/\/harmony.earthdata.nasa.gov\/schem...
(sr-local) (base) Takuyas-Laptop:super-resolution takuyakurihana$ 

"""