import json
import os

import numpy as np
import bottleneck as bn
from scipy.ndimage import generic_filter
import numpy as np
import numba as nb
from tqdm import tqdm
from scipy.ndimage import iterate_structure
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import dask.array as da
from skimage.morphology import dilation, erosion, square
from scipy.ndimage import convolve
from osgeo import gdal
import rasterio as rio

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

def dask_majority_filter(arr, size=3, iterations=1):
    def majority_func(block):
        return majority_filter_fast(block, size=size, iterations=iterations)
    
    dask_arr = da.from_array(arr, chunks=(1024, 1024))
    depth = size // 2

    return dask_arr.map_overlap(majority_func, depth=depth, boundary=0).compute()

"""Boundary Clean Filter"""
def list_boundary_clean(raster_list, iterations=1, radius=1):
    return [
        boundary_clean(raster, iterations=iterations, radius=radius)
        #for raster in raster_list
        for raster in tqdm(raster_list, desc="Boundary cleaning")
    ]

def boundary_clean(raster_array, classes=None, iterations=2, radius=3):

    if classes is None:
        classes = np.unique(raster_array[~np.isnan(raster_array)])

    result = np.copy(raster_array)
    selem = square(radius)

    for _ in range(iterations):
        for cls in classes:
            mask = result == cls
            closed = erosion(dilation(mask, selem), selem)
            result[closed] = cls 

    return result

"""Sieve Filter"""
def list_sieve_filter(array, crs, transform, iterations=1, threshold=9, connectedness=4):
    array = np.asarray(array)
    bands, height, width = array.shape
    filtered_array = np.empty_like(array, dtype="uint8")

    crs_wkt = crs.to_wkt()
    # crs_wkt = crs

    for b in tqdm(range(bands), desc="Applying Sieve Filter"):
        array_uint8 = np.nan_to_num(array[b], nan=0).astype("uint8")

        # Initialize source in-memory dataset
        src_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(transform.to_gdal())  # rasterio transform -> GDAL format
        src_ds.SetProjection(crs_wkt)
        src_ds.GetRasterBand(1).WriteArray(array_uint8)

        for _ in range(iterations):
            # Create new MEM dataset for output
            dst_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
            dst_ds.SetGeoTransform(transform.to_gdal())
            dst_ds.SetProjection(crs_wkt)

            # Apply sieve filter
            gdal.SieveFilter(
                srcBand=src_ds.GetRasterBand(1),
                maskBand=None,
                dstBand=dst_ds.GetRasterBand(1),
                threshold=threshold,
                connectedness=connectedness
            )

            # Swap for next iteration
            src_ds = dst_ds

        # Read back result
        filtered_array[b] = dst_ds.GetRasterBand(1).ReadAsArray()

    return filtered_array

#===========================================#
# Other
#===========================================#

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