import json
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
import tifffile

#===========================================#
# Processing Functions
#===========================================#

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

"""Thresholds"""
def full_threshold(raster, thresholds):
    """Apply multiple thresholds to a raster and return a list of binary arrays."""
    if isinstance(raster, xr.DataArray):
        return xarray_full_threshold_concat(raster, thresholds)
    else:
        results = []
        for t in tqdm(thresholds, desc="Applying thresholds"):
            result = threshold(raster, t)
            results.append(result)
            
        return results

def threshold(raster, threshold):
    raster = np.asarray(raster)
    return (raster > threshold).astype(raster.dtype)

def xarray_full_threshold_concat(xr_data, thresholds):
    """
    Apply multiple thresholds and return as a single DataArray with threshold dimension.
    """
    results = []
    for t in tqdm(thresholds, desc="Applying thresholds"):
        binary_result = (xr_data > t).astype(xr_data.dtype)
        results.append(binary_result)
    
    # Concatenate along a new 'threshold' dimension
    return xr.concat(results, dim='threshold').assign_coords(threshold=thresholds)

"""Combine Rasters"""
def combine_thresholded_rasters_detailed(masks_thresholded=None, param_thresholded=None):
    """
    More detailed version with explicit handling of dimensions and metadata.
    """
    masks_thresholded = masks_thresholded or []
    param_thresholded = param_thresholded or []
    
    if len(masks_thresholded) > 0 or len(param_thresholded) > 1:
        # Get threshold coordinates from first available array
        if param_thresholded:
            threshold_coords = param_thresholded[0].coords['threshold']
            template = param_thresholded[0]
        else:
            threshold_coords = masks_thresholded[0].coords['threshold']
            template = masks_thresholded[0]
        
        # Process masks (subtractive)
        if len(masks_thresholded) > 0:
            # Combine all masks using logical OR
            combined_mask_raw = masks_thresholded[0].copy() 
            for mask in masks_thresholded[1:]: # should make sure they are 2D
                combined_mask_raw =  xr.ufuncs.logical_or(combined_mask_raw, mask[0])
            
            # Invert mask (areas to keep)
            combined_mask = xr.ufuncs.logical_not(combined_mask_raw)
        else:
            # No masks - create all-ones mask
            combined_mask = xr.ones_like(template, dtype='uint32') # ???
        
        # Process parameters (multiplicative)
        if len(param_thresholded) > 1:
            param_thresholded = assign_thresholds_to_params(param_thresholded)
            combined_params = param_thresholded[0].copy()
            for param in param_thresholded[1:]:
                combined_params = combined_params * param
        else:
            combined_params = param_thresholded[0]
        
        # Apply mask
        result = combined_params * combined_mask[0]
        
        # Update metadata
        result.attrs['operation'] = 'combined_thresholded_rasters'
        result.attrs['num_masks'] = len(masks_thresholded)
        result.attrs['num_params'] = len(param_thresholded)
        
        return result
    
    return None

def assign_thresholds_to_params(param_thresholded):
    num_thresholds = len(param_thresholded[0].threshold)
    if num_thresholds > 0:
        threshold_list = list(range(1, num_thresholds + 1))
        new_param_thresholded = []
        for param in param_thresholded:
            param = param.assign_coords(threshold=threshold_list)
            new_param_thresholded.append(param)
    return new_param_thresholded

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

#===========================================#
# Other
#===========================================#

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

# def save_raster_gdal(array, crs, transform, output_path):
#     """
#     Save a raster array to a file using GDAL.
    
#     Args:
#         array (np.ndarray): The raster data to save.
#         crs (str): The coordinate reference system in WKT format.
#         transform (tuple): The affine transformation parameters.
#         output_path (str): The path where the raster will be saved.
        
#     Returns:
#         str: The path to the saved raster file.
#     """
#     driver = gdal.GetDriverByName('GTiff')
#     height, width = array.shape
#     dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
    
#     if dataset is None:
#         raise IOError(f"Could not create raster file at {output_path}")
    
#     dataset.SetGeoTransform(transform) # add error handling
#     dataset.SetProjection(crs)
    
#     dataset.GetRasterBand(1).WriteArray(array)
#     dataset.FlushCache()
    
#     return output_path

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