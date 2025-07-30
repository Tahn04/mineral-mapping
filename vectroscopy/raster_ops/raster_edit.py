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

def clip_raster(raster, mask, val=1):
        """Masks the raster data using a boolean mask or a value - optimized."""
        # Create boolean mask efficiently
        if val:
            if isinstance(mask, xr.DataArray):
                bool_mask = mask != val
            else:
                bool_mask = mask != val
        else:
            bool_mask = mask
        
        # Apply mask directly without logical_not operations
        if isinstance(raster, xr.DataArray) and isinstance(bool_mask, xr.DataArray):
            return raster.where(~bool_mask, 0)  # Use xarray's where method
        else:
            # Convert to numpy for efficient operations
            if isinstance(raster, xr.DataArray):
                raster_data = raster.values
            else:
                raster_data = raster
                
            if isinstance(bool_mask, xr.DataArray):
                mask_data = bool_mask.values
            else:
                mask_data = bool_mask
            
            return raster_data * ~mask_data