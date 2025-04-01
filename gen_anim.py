# sr-local
# 
import os
import matplotlib.colors
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import calendar

# write a function to generate animation with matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Define the custom colormap with more intermediate colors
colors = [
    (1, 1, 1),    # White (for 0 and near 0)
    (0.85, 0.85, 0.85),  # Light Grey
    (0.7, 0.7, 0.7),  # Grey
    (0.6, 0.75, 0.6),  # Pale Green
    (0.5, 0.8, 0.5),  # Light Green
    (0.25, 0.9, 0.75),  # Greenish Cyan
    (0.0, 1.0, 1.0),  # Cyan
    (0.6, 0.8, 1.0),  # Light Blueish Purple
    (0.8, 0.6, 1.0),  # Light Purple
    (0.6, 0.3, 0.8),  # Medium Purple
    (0.4, 0.0, 0.4)   # Dark Purple
]

custom_cmap = LinearSegmentedColormap.from_list("custom_precip", colors, N=256)
timestamps = []
_ , ndays = calendar.monthrange(2020, 7)
for i in range(1, ndays + 1):
    # Generate the first day of each month
    date = datetime(2020, 7, i)
    # Format the date as a string
    date_str = date.strftime("%Y-%m-%d")
    timestamps.append(date_str)

def generate_animation(images, output_path, interval=100):
    """
    Generate an animation from a list of image paths.

    Args:
        image_paths (list): List of file paths to the images.
        output_path (str): Path to save the generated animation.
        interval (int): Interval between frames in milliseconds.
    """

    
    
    fig, ax = plt.subplots(1,1, figsize=(9.5, 5.5), dpi=100, 
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    # Draw static map features once
    ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)    
    ax.imshow(images[0], cmap=custom_cmap, vmin=0, vmax=4.2, 
                extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
    ax.set_title(timestamps[0])
    #cbar = fig.colorbar(img, ax=ax, orientation='horizontal')

    def update(frame):
        """
        Update function for the animation.

        Args:
            frame (int): The current frame number.
        """
        ax.clear()
        ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)    
        img = images[frame]
        ax.imshow(img, cmap=custom_cmap, vmin=0, vmax=4.2,
                extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        ax.set_title(timestamps[frame])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

    ani = animation.FuncAnimation(
        fig, update, frames=len(images), interval=interval
    )
    ani.save(output_path, writer='imagemagick')
    plt.close(fig)

def generate_animation_diff(images, output_path, interval=100):
    """
    Generate an animation from a list of image paths.

    Args:
        image_paths (list): List of file paths to the images.
        output_path (str): Path to save the generated animation.
        interval (int): Interval between frames in milliseconds.
    """
    vmin = -2
    vmax = 2
    # Define fixed levels
    N = 10  # Number of discrete levels
    levels = np.linspace(vmin, vmax, N + 1)  # Fixed range for normalization

    cmap = matplotlib.colormaps.get_cmap('RdBu_r')
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(1,1, figsize=(9.5, 3.5), dpi=100, 
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    # Draw static map features once
    ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)    
    im1 = ax.imshow(images[0], cmap=cmap, norm=norm, 
                extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
    ax.set_title(timestamps[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar = fig.colorbar(im1, ax=ax, orientation='horizontal',fraction=0.092, pad=0.06,
                        boundaries=levels, ticks=levels)
    cbar.set_label("$log(x+1) [mm/day]$")

    def update(frame):
        """
        Update function for the animation.

        Args:
            frame (int): The current frame number.
        """
        ax.clear()
        ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)    
        img = images[frame]
        ax.imshow(img, cmap=cmap, norm=norm,
                extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        #fig.colorbar(im1, ax=ax, orientation='horizontal')
        #cbar = plt.colorbar(im1, ax=ax, orientation='horizontal') # ,
        #                #boundaries=levels, ticks=levels)
        #cbar.set_label("$log(x+1) [mm/day]$")
        ax.set_title(timestamps[frame])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
        #return im1,

    ani = animation.FuncAnimation(
        fig, update, frames=len(images), interval=interval
    )
    ani.save(output_path, writer='imagemagick')
    plt.close(fig)

def generate_animation_comparison(imerg, era5, diff, output_path, interval=100):
    """
    Generate an animation from a list of image paths.

    Args:
        image_paths (list): List of file paths to the images.
        output_path (str): Path to save the generated animation.
        interval (int): Interval between frames in milliseconds.
    """ 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    # Draw static map features once
    for ax in axes:
        ax.set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=1)    

    N0 = 9  # Number of discrete levels
    levels0 = np.linspace(0, 4.5, N0 + 1)  # Fixed range for normalization
    # IMERG
    im0 = axes[0].imshow(imerg[0], cmap=custom_cmap, vmin=0, vmax=4.2,
                          extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
    axes[0].set_title(f"IMERG {timestamps[0]}")
    cbar = fig.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04,
                        boundaries=levels0, ticks=levels0)
    cbar.set_label("$log(x+1) [mm/day]$")

    # ERA5
    im1 = axes[1].imshow(era5[0], cmap=custom_cmap, vmin=0, vmax=4.2,
                          extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
    axes[1].set_title(f"ERA5 {timestamps[0]}")
    cbar = fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04,
                        boundaries=levels0, ticks=levels0)
    cbar.set_label("$log(x+1) [mm/day]$")

    # Difference
    vmin = -2
    vmax = 2
    cmap = matplotlib.colormaps.get_cmap('RdBu_r')
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # Define fixed levels
    N = 10  # Number of discrete levels
    levels = np.linspace(vmin, vmax, N + 1)  # Fixed range for normalization
    
    im2 = axes[2].imshow(diff[0], cmap=cmap, norm=norm,
                          extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
    axes[2].set_title(f"IMERG - ERA5 {timestamps[0]}")
    cbar = fig.colorbar(im2, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.04,
                        boundaries=levels, ticks=levels)
    cbar.set_label("$log(x+1) [mm/day]$")
    plt.tight_layout()

    
    def update(frame):
        """
        Update function for the animation.

        Args:
            frame (int): The current frame number.
        """
        #ax.clear()
        for i in range(3):
            img = [imerg[frame], era5[frame], diff[frame]][i]
            axes[i].set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
            axes[i].add_feature(cfeature.COASTLINE, linewidth=1)
            if i == 2:
                axes[i].imshow(img, cmap=cmap, norm=norm,
                extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
            else:
                axes[i].imshow(img, cmap=custom_cmap, vmin=0, vmax=4.2,
                extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
            axes[i].set_title(timestamps[frame])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            axes[i].axis('off')
        axes[0].set_title(f"IMERG {timestamps[frame]}")
        axes[1].set_title(f"ERA5 {timestamps[frame]}")     
        axes[2].set_title(f"IMERG - ERA5 {timestamps[frame]}")
        plt.tight_layout()
    ani = animation.FuncAnimation(  
        fig, update, frames=len(era5), interval=interval
    )
    ani.save(output_path, writer='imagemagick')
    plt.close(fig)

def logtransform(x, m2mm_per_day=False):
    if m2mm_per_day:
        x *= 1000
        x[x<0.25] = 0 # 0.25 mm/day ~=0
    return np.log1p(x)

def read_imerg(    
    image_dir = "./imerg-2020-july-1d-3600x1800.zarr",
    m2mm_per_day = False,
):
    # Get a list of image file paths
    ds = xr.open_zarr(image_dir).precipitation
    # update latitude
    ds = ds.sortby("lat", ascending=False)
    # update longitude
    lons = ds.lon.values
    lons[lons < 0] += 360
    ds = ds.assign_coords(lon=lons)
    ds = ds.sortby("lon", ascending=True)

    # check updated latlon
    print(ds)

    # Get the precip (mm/day) image
    images = ds.values
    images = np.swapaxes(images, -1, -2) # Reverse the order of the last two axes
    images = logtransform(images, m2mm_per_day)
    return images

def read_era5(filepath="/Users/takuyakurihana/Projects/super-resolution/era5-2020-july.nc",
              m2mm_per_day = True):
    # Get a list of image file paths
    ds = xr.open_dataset(filepath).tp
    # update latitude
    ds = ds.sortby("latitude", ascending=False)
    # update longitude
    lons = ds.longitude.values
    lons[lons < 0] += 360
    ds = ds.assign_coords(longitude=lons)
    ds = ds.sortby("longitude", ascending=True)
    # sum up the precip for every day
    ds = ds.resample(valid_time="1D").sum(dim="valid_time")

    # check updated latlon
    print(ds)
    try:
        ds.to_netcdf("./era5-2020-july-1d-1440x721.nc", mode="w")
    except FileExistsError:
        print("File already exists")
        pass

    # Get the precip (mm/day) image
    images = ds.values
    images = logtransform(images, m2mm_per_day)
    return images


def era5_test(filepath):
    # Get a list of image file paths
    ds = xr.open_dataset(filepath)
    # update latitude
    ds = ds.sortby("latitude", ascending=False)
    # update longitude
    lons = ds.longitude.values
    lons[lons < 0] += 360
    ds = ds.assign_coords(longitude=lons)
    ds = ds.sortby("longitude", ascending=True)
    #ds = ds.sel(valid_time="2020-07-01")
    ds = ds.resample(valid_time="1D").sum(dim="valid_time")
    print(ds)
    return ds

if __name__ == "__main__":
    
    #output_path = "animation_res.gif"
    #output_path = "animation_era5.gif"
    #output_path = "animation.gif"

    ## IMERG
    #imerg = read_imerg()
    #print("Check shape and type", imerg.shape, type(imerg))

    # Generate the imerg animation
    #output_path = "animation_imerg.gif"
    #generate_animation(imerg, output_path, interval=200)
    
    ## ERA5
    #if os.path.exists("./era5-2022-july-1d-1440x721.nc"):
    #    print('Use preprocessed era5')
    #    era5 = xr.open_dataset("./era5-2022-july-1d-1440x721.nc")
    #    era5 = logtransform(era5.tp.values,True)   
    #else:
    #    print('Start reading era5')
    #    era5 = read_era5()
    #print("Check shape and type", era5.shape, type(era5))
    ## Generate the era5 animation
    #output_path = "animation_era5.gif"
    #generate_animation(era5, output_path, interval=200)


    ## IMERG - ERA5
    imerg_interp = xr.open_dataset("./imerg-2020-july-1d-1440x721.nc")
    imerg_interp = logtransform(imerg_interp.precipitation.values,False)   
    imerg_interp = np.swapaxes(imerg_interp, -1, -2) # Reverse the order of the last two axes
    
    era5 = xr.open_dataset("./era5-2020-july-1d-1440x721.nc")
    #era5 = era5_test("./era5-2020-july.nc") # read from original
    era5 = logtransform(era5.tp.values,True)   
    print("Shape check IMERG and ERA5: ", imerg_interp.shape, era5.shape)

    #assert imerg_interp.shape == era5.shape, f"Shape mismatch between IMERG {imerg_interp.shape} and ERA5 {era5.shape}"
    #diff = imerg_interp[:1] - era5
    diff = imerg_interp - era5
    print("Check shape and type", diff.shape, type(diff))

    # Create histogram of the difference
    if not os.path.exists("./histogram_diff.png"):
        print('Start generating histogram')
        plt.figure(figsize=(10, 6))
        plt.hist(diff.flatten(), bins=100, color='blue', alpha=0.7)
        plt.xlabel('Difference (mm/day)')
        plt.ylabel('Frequency')
        plt.title('Histogram of IMERG - ERA5 Difference')
        plt.grid()
        plt.savefig("./histogram_diff.png")
        plt.close()
    
    # Generate a figure with IMERG, ERA5, and the difference in one row and three columns
    if not os.path.exists("./comparison_figure1.png"):
        print('Start generating comparison figure')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        # IMERG
        axes[0].set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        axes[0].add_feature(cfeature.COASTLINE, linewidth=1)
        im0 = axes[0].imshow(imerg_interp[1], cmap=custom_cmap, vmin=0, vmax=4.2,
                              extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        axes[0].set_title("IMERG")
        fig.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04)

        # ERA5
        axes[1].set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        axes[1].add_feature(cfeature.COASTLINE, linewidth=1)
        im1 = axes[1].imshow(era5[1], cmap=custom_cmap, vmin=0, vmax=4.2,
                              extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        axes[1].set_title("ERA5")
        fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04)

        # Difference
        axes[2].set_extent([0, 360, -90, 90], crs=ccrs.PlateCarree())
        axes[2].add_feature(cfeature.COASTLINE, linewidth=1)
        im2 = axes[2].imshow(diff[1], cmap='RdBu_r', norm=matplotlib.colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3),
                              extent=[0, 360, -90, 90], transform=ccrs.PlateCarree())
        axes[2].set_title("IMERG - ERA5")
        fig.colorbar(im2, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("./comparison_figure1.png")
        plt.close()
    
    # Generate the imerg - era5 animation
    output_path = "animation_diff.gif"
    generate_animation_diff(diff, output_path, interval=200)

    output_path = "animation_diff_comp.gif"
    generate_animation_comparison(imerg_interp, era5, diff, output_path, interval=200)