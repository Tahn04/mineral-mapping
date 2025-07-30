import os
from affine import Affine
import numpy as np
import bottleneck as bn
import numpy as np
from tqdm import tqdm
from scipy.ndimage import iterate_structure
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import dask.array as da
from skimage.morphology import dilation, erosion, footprint_rectangle
from scipy.ndimage import convolve
# from osgeo import gdal
import rasterio as rio
from scipy.ndimage import binary_opening
from skimage.morphology import square
from scipy import ndimage
import xarray as xr
from rasterio.features import sieve

def xarray_to_array(xr_data):
    """Convert an Xarray DataArray to a NumPy array."""
    if isinstance(xr_data, xr.DataArray):
        return [np.asarray(xr_data.sel(threshold=t).values) for t in xr_data.coords['threshold'].values]

def array_to_xarray(array_list):
    """Convert a list of NumPy arrays to an Xarray DataArray."""
    if all(isinstance(arr, np.ndarray) for arr in array_list):
        return xr.DataArray(data=array_list)

def label_clusters(binary_raster, connectivity=1):
    """
    Labels connected regions of 1s in a binary raster.
    
    Parameters:
    - binary_raster (np.ndarray): 2D array of 0s and 1s.
    - connectivity (int): 1 for 4-connected, 2 for 8-connected (diagonals included).
    
    Returns:
    - labeled (np.ndarray): Same shape as input, with unique labels for each cluster.
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled = ndimage.label(binary_raster, structure=structure)[0]
    return labeled

def get_raster_thresholds(raster, thresholds=['75p', '85p', '95p']):
    """
    Calculate thresholds for a raster based on specified percentiles.
    
    Args:
        raster (numpy.ndarray): Input raster data.
        percentile_list (list): List of percentiles to calculate thresholds for.
        
    Returns:
        dict: Dictionary with percentile values as keys and corresponding threshold values as values.
    """
    temp_thresholds = []
    for t in thresholds:
        if isinstance(t, str) and t.endswith('p'):
            p = float(t[:-1])
            temp_thresholds.append(np.nanpercentile(raster, p).round(4) )
        elif isinstance(t, (int, float)):
            temp_thresholds.append(t)
        else:
            raise ValueError(f"Invalid threshold format: {t}. Must be a number or a string ending with 'p'.")
    
    return temp_thresholds

def show_raster(raster, cmap='gray', title=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    img = plt.imshow(raster, cmap=cmap)
    plt.colorbar(img, label='Value')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def save_raster(raster, output_path, file_name, profile):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    full_path = os.path.join(output_path, file_name)
    raster = np.asarray(raster)
    with rio.open(full_path, 'w', **profile) as dst:
        # If raster is 2D, add a band axis
        if raster.ndim == 2:
            dst.write(raster, 1)
        else:
            dst.write(raster)

def save_raster_fast_rasterio(array, crs, transform, filepath):
    """Save with rasterio using fast settings"""
    if transform is not None and not isinstance(transform, Affine):
        transform = Affine(transform[1], transform[2], transform[0],
                         transform[4], transform[5], transform[3])
    with rio.open(
        filepath,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,  # Number of bands
        dtype=array.dtype,  # Data type
        crs=crs,
        transform=transform,
        # compress='lzw',  # Fast compression
        tiled=True
    ) as dst:
        dst.write(array, 1)

def save_raster_np_array(array, crs, transform, output_path):
    """
    Save a raster array to a file using numpy.
    """
    np.savez_compressed(output_path, array=array, crs=crs, transform=transform)