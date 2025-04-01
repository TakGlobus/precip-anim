import xarray as xr
import os
import glob
# Define the directory containing NetCDF files
input_dir = "/Users/takuyakurihana/Projects/super-resolution/datasets/IMERG"
output_zarr = "./imerg-2022-july-1d-3600x1800.zarr" # mm/day

# Get a list of all NetCDF files in the directory
netcdf_files = glob.glob(os.path.join(input_dir, "*.nc4*"))

# Open multiple NetCDF files as a single xarray dataset
ds = xr.open_mfdataset(netcdf_files, combine="by_coords")

# Save the dataset to Zarr format
ds.to_zarr(output_zarr)