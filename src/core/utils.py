import os

import rasterio as rio
from rasterio import features
import numpy as np
import pandas as pd
import geopandas as gpd
import bottleneck as bn
from shapely.geometry import shape
from scipy.ndimage import generic_filter # GET RID OF THIS
from scipy.signal import convolve2d

# Raster Operations
def show_raster(raster, cmap='gray', title=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    img = plt.imshow(raster, cmap=cmap)
    plt.colorbar(img, label='Value')  # Adds the colorbar with label
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def threshold(raster, threshold):
    raster = np.asarray(raster)
    return (raster > threshold).astype(raster.dtype)

def mask(raster, mask):
    return raster * mask

def open_raster(raster_path):
    return rio.open(raster_path) 

def open_raster_band(raster, band_number):
    return raster.read(band_number, masked=True).filled(np.nan)

# Filters
from scipy.ndimage import median_filter

# fast but not as accurate
# def median_kernel_filter(raster, size=3, iterations=1): 
#     for _ in range(iterations):
#         raster = median_filter(raster, size=size)
#     return raster

# More accurate but slower
def median_kernel_filter(raster, size=3, iterations=1):
    def bn_nanmedian(arr):
        return bn.nanmedian(arr)
    for _ in range(iterations):
        raster = generic_filter(raster, bn_nanmedian, size=3)
    return raster

# def majority_filter(raster):
#     def filter_func(values):
#         count_true = np.sum(values)
#         return count_true > (len(values) / 2)
#     return generic_filter(raster, filter_func, size=(3, 3), mode='nearest')


def majority_filter(binary_array, window_size=3, iterations=1):
    def mfilter(binary_array, window_size=3):
        kernel = np.ones((window_size, window_size), dtype=np.uint8)

        binary_array = np.nan_to_num(binary_array, nan=0).astype(np.uint8)
        
        count = convolve2d(binary_array, kernel, mode='same', boundary='symm')
        
        threshold = (window_size * window_size) // 2
        return (count > threshold).astype(np.uint8)
    for _ in range(iterations):
        binary_array = mfilter(binary_array, window_size=window_size)
    return binary_array

def boundary_clean(raster_array, classes=None, iterations=1, radius=1):
    """
    Perform smoothing on raster zones using morphological operations.

    Parameters:
    - raster_array: 2D numpy array of the raster (categorical zones)
    - classes: list of class values to smooth (optional)
    - iterations: number of smoothing iterations
    - radius: neighborhood radius (like a moving window size)

    Returns:
    - Smoothed array
    """
    from skimage.morphology import disk, dilation, erosion, square

    if classes is None:
        classes = np.unique(raster_array[~np.isnan(raster_array)])

    result = np.copy(raster_array)
    selem = square(radius)

    for _ in range(iterations):
        for cls in classes:
            mask = result == cls
            # Close operation: dilation followed by erosion
            closed = erosion(dilation(mask, selem), selem)
            result[closed] = cls  # Reassign class where zone expands

    return result

# Vector Operations
def vectorize(mask_array, value, transform, crs):
    """
    Vectorize a boolean mask array, assigning a value to the polygons.

    Parameters:
        mask_array (np.ndarray): Boolean mask array.
        value (int or float): Value to assign to all polygons.
        transform (Affine): Affine transform for the array.
        crs: Coordinate reference system.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of polygons.
    """
    # Convert boolean mask to uint8 for rasterio.features.shapes
    mask_uint8 = mask_array.astype(np.uint8)
    shapes = features.shapes(mask_uint8, transform=transform) # should be rio.features.shapes
    polygons = [ 
        {"geometry": shape(geom), "value": value}
        for geom, val in shapes if val == 1
    ]
    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    # gdf = gdf.drop(columns="value")
    return gdf

def merge_polygons(gdf, tolerance=0.01):
    """
    Merge polygons in a GeoDataFrame.

    Parameters:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with polygons.
        tolerance (float): Tolerance for merging.

    Returns:
        gpd.GeoDataFrame: Merged GeoDataFrame.
    """
    return pd.concat(gdf, ignore_index=True)
    # merged = gdf.dissolve(by="value", tolerance=tolerance)
    # return merged.reset_index(drop=True)

def show_polygons(gdf, title=None):
    """
    Display polygons using matplotlib.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame with polygons.
        title (str): Title for the plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(ax=ax, color='blue', edgecolor='black')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def save_raster(raster, output_path, profile):
    """
    Save a raster dataset to a file.

    Parameters:
        raster (numpy.ndarray): Raster data to save.
        output_path (str): Path to save the raster.
        profile (dict): Metadata for the raster.
    """
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(raster, 1)

def save_shapefile(gdf, output_path, file_name):
    """
    Save a GeoDataFrame to a shapefile.

    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame to save.
        output_path (str): Directory to save the shapefile.
        file_name (str): Name of the shapefile (without extension).
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gdf.to_file(os.path.join(output_path, file_name), driver='ESRI Shapefile')