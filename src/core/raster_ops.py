import json
import os

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
from osgeo import gdal
import rasterio as rio
from scipy.ndimage import binary_opening
from skimage.morphology import square
from scipy import ndimage

#===========================================#
# Processing Functions
#===========================================#

"""Median Filter"""
def dask_nanmedian_filter(arr, window_size=3, iterations=1):
    dask_arr = da.from_array(arr, chunks=(1024, 1024))  # Adjust chunk size as needed

    for _ in tqdm(range(iterations), desc="Applying Dask nanmedian filter"):
        dask_arr = dask_arr.map_overlap(
            nanmedian_2d,
            window_size=window_size,
            depth=window_size // 2,
            boundary=np.nan,
            dtype=arr.dtype
        )

    return dask_arr.compute()

def nanmedian_2d(x, window_size):
    """Apply 2D nanmedian filter to a NumPy array with given window size."""
    pad = window_size // 2
    padded = np.pad(x, pad, mode='constant', constant_values=np.nan)

    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
    windows = windows.reshape(windows.shape[0], windows.shape[1], -1)

    return bn.nanmedian(windows, axis=2)

"""Thresholds"""
def full_threshold(raster, thresholds):
    """Apply multiple thresholds to a raster and return a list of binary arrays."""
    results = []
    for t in tqdm(thresholds, desc="Applying thresholds"):
        result = threshold(raster, t)
        results.append(result)
        
    return results

def threshold(raster, threshold):
    raster = np.asarray(raster)
    return (raster > threshold).astype(raster.dtype)


"""Majority Filter"""
def list_majority_filter(raster_list, iterations=1, size=3):
    return [
        majority_filter_fast(raster, size=size, iterations=iterations)
        #for raster in raster_list
        for raster in tqdm(raster_list, desc="Applying majority filter")
    ]

def majority_filter_fast(binary_array, size=3, iterations=1):
    kernel = np.ones((size, size), dtype=np.uint8)
    array = np.nan_to_num(binary_array, nan=0).astype(np.uint8)
    threshold = (size * size) // 2

    for _ in range(iterations):
        count = convolve(array, kernel, mode='mirror')
        array = (count > threshold).astype(np.uint8)

    return array

def dask_list_majority_filter(raster_list, iterations=1, size=3, chunk_size=(1024, 1024)):
    """Apply majority filter to a list of rasters using Dask for parallelization."""
    return [
        dask_majority_filter(raster, size=size, iterations=iterations, chunk_size=chunk_size)
        for raster in tqdm(raster_list, desc="Applying Dask majority filter")
    ]

def dask_majority_filter(binary_array, size=3, iterations=1, chunk_size=(1024, 1024)):
    """Apply majority filter using Dask for memory efficiency and potential speed gains."""
    # Convert to dask array
    dask_arr = da.from_array(binary_array, chunks=chunk_size)
    
    # Apply majority filter iterations
    for _ in range(iterations):
        dask_arr = dask_arr.map_overlap(
            majority_filter_kernel,
            size=size,
            depth=size // 2,
            boundary='reflect',  # Use reflect instead of nan for binary data
            dtype=np.uint8
        )
    
    return dask_arr.compute()

def majority_filter_kernel(x, size):
    """Apply majority filter kernel to a chunk."""
    kernel = np.ones((size, size), dtype=np.uint8)
    array = np.nan_to_num(x, nan=0).astype(np.uint8)
    threshold = (size * size) // 2
    
    count = convolve(array, kernel, mode='mirror')
    return (count > threshold).astype(np.uint8)


"""Boundary Clean Filter"""
def list_boundary_clean(raster_list, iterations=1, radius=1):
    return [
        boundary_clean(raster, iterations=iterations, radius=radius)
        #for raster in raster_list
        for raster in tqdm(raster_list, desc="Boundary cleaning")
    ]

def boundary_clean(raster_array, iterations=2, radius=3):
    """
    Smooth binary raster boundaries similar to ArcGIS Boundary Clean tool.
    
    Parameters:
    - raster_array (np.ndarray): Binary array (1 = feature, 0 = background)
    - iterations (int): How many expand-shrink cycles to perform
    - radius (int): Structuring element size (larger = more aggressive smoothing)
    
    Returns:
    - np.ndarray: Smoothed binary raster
    """
    result = np.copy(raster_array).astype(np.uint8)
    selem = footprint_rectangle((radius, radius))
    

    for _ in range(iterations):
        expanded = dilation(result, selem)
        result = erosion(expanded, selem)

    return result

"""Sieve Filter"""
def list_sieve_filter(array_list, crs, transform, iterations=1, threshold=9, connectedness=4):
    filtered_array = []

    for array in tqdm(array_list, desc="Applying Sieve Filter"):
        height, width = array.shape
        array_uint8 = np.nan_to_num(array, nan=0).astype("uint8")

        src_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(transform)
        src_ds.SetProjection(crs)
        src_ds.GetRasterBand(1).WriteArray(array_uint8)

        for _ in range(iterations):
            dst_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
            dst_ds.SetGeoTransform(transform)
            dst_ds.SetProjection(crs)

            gdal.SieveFilter(
                srcBand=src_ds.GetRasterBand(1),
                maskBand=None,
                dstBand=dst_ds.GetRasterBand(1),
                threshold=threshold,
                connectedness=connectedness
            )
            src_ds = dst_ds

        filtered_array.append(dst_ds.GetRasterBand(1).ReadAsArray())

    return filtered_array

"""Binary Opening"""
def list_binary_opening(raster_list, iterations, size):
    """
    Apply binary opening to a list of rasters.
    
    Args:
        raster_list (list of np.ndarray): List of binary rasters.
        
    Returns:
        list: List of rasters after applying binary opening.
    """
    return [
        _binary_opening(raster, iterations=iterations, size=size)
        for raster in tqdm(raster_list, desc="Applying binary opening")
    ]

def _binary_opening(raster, iterations, size):
    """
    Apply binary opening to a single raster.
    
    Args:
        raster (np.ndarray): Input binary raster.
        structure (np.ndarray): Structuring element for the opening operation.
        
    Returns:
        np.ndarray: Raster after applying binary opening.
    """
    if not isinstance(raster, np.ndarray):
        raise ValueError("Input raster must be a NumPy array.")
    
    structure=footprint_rectangle((size, size))
    
    return binary_opening(raster, structure=structure, iterations=iterations)

#===========================================#
# Other
#===========================================#

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

def open_raster(raster_path):
    return rio.open(raster_path) 

def open_raster_band(raster, band_number):
    return raster.read(band_number, masked=True).filled(np.nan)

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

def save_raster_gdal(array, crs, transform, output_path):
    """
    Save a raster array to a file using GDAL.
    
    Args:
        array (np.ndarray): The raster data to save.
        crs (str): The coordinate reference system in WKT format.
        transform (tuple): The affine transformation parameters.
        output_path (str): The path where the raster will be saved.
        
    Returns:
        str: The path to the saved raster file.
    """
    driver = gdal.GetDriverByName('GTiff')
    height, width = array.shape
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    
    if dataset is None:
        raise IOError(f"Could not create raster file at {output_path}")
    
    dataset.SetGeoTransform(transform) # add error handling
    dataset.SetProjection(crs)
    
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    
    return output_path