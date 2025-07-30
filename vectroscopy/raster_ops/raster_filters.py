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

"""Median Filter"""
def dask_nanmedian_filter(data, window_size=3, iterations=1):
    """
    Apply nanmedian filter to either numpy array or Xarray DataArray.
    Returns lazy Dask-backed DataArray if input is Xarray, otherwise numpy array.
    """
    # Handle both numpy arrays and Xarray DataArrays
    if isinstance(data, xr.DataArray):
        # Work with the underlying Dask array (or convert to Dask if numpy-backed)
        if data.chunks is None:
            # If not chunked, create chunks
            dask_arr = da.from_array(data.values, chunks=(1024, 1024))
        else:
            # Already chunked
            dask_arr = data.data
        
        # Apply the filter iterations
        for _ in tqdm(range(iterations), desc="Applying Xarray nanmedian filter"):
            dask_arr = dask_arr.map_overlap(
                nanmedian_2d,
                window_size=window_size,
                depth=window_size // 2,
                boundary=np.nan,
                dtype=dask_arr.dtype
            )

        # Return as lazy Xarray DataArray
        return xr.DataArray(
            dask_arr,  # Keep as Dask array (lazy)
            coords=data.coords,
            dims=data.dims,
            attrs=data.attrs
        )
    else:
        # Original behavior for numpy arrays
        # dask_arr = da.from_array(data, chunks=(1024, 1024))
        dask_arr = data

        for _ in tqdm(range(iterations), desc="Applying Dask nanmedian filter"):
            dask_arr = dask_arr.map_overlap(
                nanmedian_2d,
                window_size=window_size,
                depth=window_size // 2,
                boundary=np.nan,
                dtype=data.dtype
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
            boundary='reflect',
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
# def list_sieve_filter(array_list, crs, transform, iterations=1, threshold=9, connectedness=4):
#     filtered_array = []

#     for array in tqdm(array_list, desc="Applying Sieve Filter"):
#         height, width = array.shape
#         array_uint8 = np.nan_to_num(array, nan=0).astype("uint8")

#         src_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
#         src_ds.SetGeoTransform(transform)
#         src_ds.SetProjection(crs)
#         src_ds.GetRasterBand(1).WriteArray(array_uint8)

#         for _ in range(iterations):
#             dst_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
#             dst_ds.SetGeoTransform(transform)
#             dst_ds.SetProjection(crs)

#             gdal.SieveFilter(
#                 srcBand=src_ds.GetRasterBand(1),
#                 maskBand=None,
#                 dstBand=dst_ds.GetRasterBand(1),
#                 threshold=threshold,
#                 connectedness=connectedness
#             )
#             src_ds = dst_ds

#         filtered_array.append(dst_ds.GetRasterBand(1).ReadAsArray())

#     return filtered_array

def list_sieve_filter_rio(array_list, iterations=1, threshold=9, connectedness=4):
    """
    Apply sieve filter to a list of arrays using rasterio.
    
    Parameters:
    - array_list: List of 2D numpy arrays to filter
    - crs: Coordinate reference system (not used by rasterio sieve but kept for compatibility)
    - transform: Affine transform (not used by rasterio sieve but kept for compatibility) 
    - iterations: Number of sieve iterations to apply
    - threshold: Minimum number of connected pixels to keep
    - connectedness: Pixel connectivity (4 or 8)
    
    Returns:
    - List of filtered arrays
    """
    filtered_array = []

    for array in tqdm(array_list, desc="Applying Sieve Filter"):
        # Convert to uint8 and handle NaN values
        array_uint8 = np.nan_to_num(array, nan=0).astype("uint8")
        
        # Apply sieve filter for specified iterations
        filtered = array_uint8.copy()
        for _ in range(iterations):
            filtered = sieve(
                source=filtered,
                size=threshold,
                connectivity=connectedness
            )
            filtered_array.append(filtered)

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